#!/usr/bin/env python3
"""
Vertex OCR Script
=================
使用 Vertex AI 的 Gemini 模型处理扫描版 PDF，支持实时模式和 Batch 模式。

Batch 模式工作流：
1. 本地将 PDF 切页为 PNG
2. 上传每页图片到 GCS
3. 生成 Vertex Batch 所需的 JSONL 请求文件并上传到 GCS
4. 提交 Vertex Batch Job
5. 轮询状态并从 GCS 拉回结果
6. 合并为 Markdown
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    from google import genai
    from google.cloud import storage
    from google.genai import types
    from pdf2image import convert_from_path
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    print(f"缺少依赖: {exc}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"vertex_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _optional_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def _safe_ascii_name(name: str) -> str:
    if name.isascii():
        return name
    import hashlib

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"pdf_{digest}"


def _normalize_gcs_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _split_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"非法 GCS URI: {gcs_uri}")
    remainder = gcs_uri[5:]
    bucket, _, blob = remainder.partition("/")
    return bucket, blob


def _join_gcs_uri(bucket: str, *parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    suffix = "/".join(cleaned)
    return f"gs://{bucket}/{suffix}" if suffix else f"gs://{bucket}"


def _state_name(state: object) -> str:
    if hasattr(state, "name"):
        return str(state.name)
    return str(state or "")


def _extract_text_from_response(response: Dict) -> str:
    candidates = response.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "\n".join(texts).strip()


@dataclass
class Config:
    base_dir: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    status_file: Path
    page_cache_dir: Path
    batch_requests_dir: Path
    batch_results_dir: Path
    batch_status_file: Path
    vertex_api_key: str
    vertex_project: str
    vertex_location: str
    gcs_bucket: str
    gcs_prefix: str
    model_name: str
    dpi: int
    max_retries: int
    retry_delay: float
    request_delay: float
    poll_interval: int
    max_active_batch_jobs: Optional[int]
    primary_language: str
    ocr_prompt: str

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()

        base_dir = Path(__file__).parent
        primary_language = os.getenv("PRIMARY_LANGUAGE", "Arabic")
        model_name = (
            os.getenv("VERTEX_MODEL_NAME", "").strip()
            or os.getenv("MODEL_NAME", "").strip()
            or "gemini-2.5-flash-lite"
        )

        return cls(
            base_dir=base_dir,
            input_dir=base_dir / "input",
            output_dir=base_dir / "output",
            logs_dir=base_dir / "logs",
            status_file=base_dir / "logs" / "vertex_processing_status.csv",
            page_cache_dir=base_dir / "logs" / "vertex_page_cache",
            batch_requests_dir=base_dir / "logs" / "vertex_batch_requests",
            batch_results_dir=base_dir / "logs" / "vertex_batch_results",
            batch_status_file=base_dir / "logs" / "vertex_batch_jobs.json",
            vertex_api_key=os.getenv("VERTEX_API_KEY", "").strip(),
            vertex_project=(
                os.getenv("VERTEX_PROJECT", "").strip()
                or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
            ),
            vertex_location=(
                os.getenv("VERTEX_LOCATION", "").strip()
                or os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
                or "global"
            ),
            gcs_bucket=os.getenv("VERTEX_GCS_BUCKET", "").strip(),
            gcs_prefix=_normalize_gcs_prefix(
                os.getenv("VERTEX_GCS_PREFIX", "gemini-batch-by-Claude/vertex-ocr")
            ),
            model_name=model_name,
            dpi=int(os.getenv("PDF_DPI", "200")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "5")),
            request_delay=float(os.getenv("REQUEST_DELAY", "1.5")),
            poll_interval=int(os.getenv("POLL_INTERVAL", "30")),
            max_active_batch_jobs=_optional_int_env("MAX_ACTIVE_BATCH_JOBS"),
            primary_language=primary_language,
            ocr_prompt=os.getenv("OCR_PROMPT", cls.default_prompt(primary_language)),
        )

    @staticmethod
    def default_prompt(primary_language: str) -> str:
        return f"""You are an expert OCR system specialized in {primary_language} document processing.

Task: Convert this scanned document page to Markdown format.

Requirements:
1. Preserve the EXACT original text - do not translate or modify any content
2. Maintain the document structure (headings, paragraphs, lists, tables)
3. For Arabic/Hebrew text, preserve right-to-left (RTL) formatting
4. Use appropriate Markdown syntax:
   - # ## ### for headings
   - - or * for bullet lists
   - 1. 2. 3. for numbered lists
   - | for tables
   - > for quotes
5. If text is unclear, mark with [unclear: best guess]
6. Preserve any numbers, dates, and special characters exactly
7. Do not add any commentary or explanations

Output the Markdown content only, nothing else."""

    def validate(self, *, use_batch_api: bool) -> bool:
        if not self.vertex_project:
            logging.error("缺少 Vertex 项目配置：请设置 GOOGLE_CLOUD_PROJECT 或 VERTEX_PROJECT")
            return False

        for path in [
            self.input_dir,
            self.output_dir,
            self.logs_dir,
            self.page_cache_dir,
            self.batch_requests_dir,
            self.batch_results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        if use_batch_api and not self.gcs_bucket:
            logging.error("Batch 模式需要设置 VERTEX_GCS_BUCKET")
            return False

        return True


class StatusManager:
    def __init__(self, status_file: Path, page_cache_dir: Path):
        self.status_file = status_file
        self.page_cache_dir = page_cache_dir
        self.status: Dict[str, Dict[str, str]] = {}
        self._load()

    def _key(self, pdf_file: str, page_num: int) -> str:
        return f"{pdf_file}:{page_num}"

    def _load(self) -> None:
        if not self.status_file.exists():
            return

        with open(self.status_file, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.status[self._key(row["pdf_file"], int(row["page_num"]))] = row

    def _save(self) -> None:
        fieldnames = [
            "pdf_file",
            "page_num",
            "status",
            "timestamp",
            "error",
            "page_file",
        ]
        with open(self.status_file, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.status.values())

    def _cache_path(self, pdf_file: str, page_num: int) -> Path:
        pdf_dir = self.page_cache_dir / Path(pdf_file).stem
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return pdf_dir / f"page_{page_num:04d}.md"

    def is_completed(self, pdf_file: str, page_num: int) -> bool:
        row = self.status.get(self._key(pdf_file, page_num), {})
        page_file = row.get("page_file", "")
        return row.get("status") == "completed" and bool(page_file) and Path(page_file).exists()

    def mark_completed(self, pdf_file: str, page_num: int, content: str) -> None:
        cache_path = self._cache_path(pdf_file, page_num)
        cache_path.write_text(content, encoding="utf-8")
        self.status[self._key(pdf_file, page_num)] = {
            "pdf_file": pdf_file,
            "page_num": str(page_num),
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "error": "",
            "page_file": str(cache_path),
        }
        self._save()

    def mark_failed(self, pdf_file: str, page_num: int, error: str) -> None:
        previous = self.status.get(self._key(pdf_file, page_num), {})
        self.status[self._key(pdf_file, page_num)] = {
            "pdf_file": pdf_file,
            "page_num": str(page_num),
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": error[:500],
            "page_file": previous.get("page_file", ""),
        }
        self._save()

    def get_page_content(self, pdf_file: str, page_num: int) -> Optional[str]:
        row = self.status.get(self._key(pdf_file, page_num), {})
        page_file = row.get("page_file", "")
        if not page_file:
            return None
        path = Path(page_file)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def get_page_error(self, pdf_file: str, page_num: int) -> Optional[str]:
        row = self.status.get(self._key(pdf_file, page_num), {})
        error = row.get("error", "")
        return error or None

    def get_stats(self) -> Dict[str, int]:
        stats = {"completed": 0, "failed": 0, "total": len(self.status)}
        for item in self.status.values():
            if item["status"] == "completed":
                stats["completed"] += 1
            elif item["status"] == "failed":
                stats["failed"] += 1
        return stats


class BatchJobManager:
    TERMINAL_STATES = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    ACTIVE_STATES = {
        "JOB_STATE_QUEUED",
        "JOB_STATE_PENDING",
        "JOB_STATE_RUNNING",
        "JOB_STATE_CANCELLING",
        "JOB_STATE_PAUSED",
    }

    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.jobs: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.status_file.exists():
            return
        self.jobs = json.loads(self.status_file.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.status_file.write_text(
            json.dumps(self.jobs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_job(self, pdf_file: str) -> Optional[Dict[str, str]]:
        return self.jobs.get(pdf_file)

    def upsert_job(self, pdf_file: str, job_data: Dict[str, str]) -> None:
        self.jobs[pdf_file] = job_data
        self._save()

    def update_job(self, pdf_file: str, **updates: str) -> None:
        if pdf_file not in self.jobs:
            return
        self.jobs[pdf_file].update(updates)
        if updates.get("status") in self.TERMINAL_STATES:
            self.jobs[pdf_file]["completed_at"] = datetime.now().isoformat()
        self._save()

    def get_active_jobs(self) -> List[Tuple[str, Dict[str, str]]]:
        return [
            (pdf_file, job)
            for pdf_file, job in self.jobs.items()
            if job.get("status") in self.ACTIVE_STATES
        ]

    def get_stats(self) -> Dict[str, int]:
        stats = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0, "total": 0}
        for job in self.jobs.values():
            stats["total"] += 1
            status = job.get("status", "")
            if status in {"JOB_STATE_QUEUED", "JOB_STATE_PENDING"}:
                stats["pending"] += 1
            elif status in {"JOB_STATE_RUNNING", "JOB_STATE_CANCELLING", "JOB_STATE_PAUSED"}:
                stats["running"] += 1
            elif status == "JOB_STATE_SUCCEEDED":
                stats["succeeded"] += 1
            elif status in self.TERMINAL_STATES - {"JOB_STATE_SUCCEEDED"}:
                stats["failed"] += 1
        return stats

    def all_jobs(self) -> List[Tuple[str, Dict[str, str]]]:
        return list(self.jobs.items())

    def sync_from_remote(self, client: genai.Client, logger: logging.Logger) -> Dict[str, int]:
        updated = {"synced": 0, "errors": 0}
        for pdf_file, job in self.jobs.items():
            status = job.get("status", "")
            if status in self.TERMINAL_STATES:
                continue

            try:
                remote = client.batches.get(name=job["job_name"])
                remote_state = _state_name(remote.state)
                next_updates: Dict[str, str] = {"status": remote_state}
                if remote.dest and remote.dest.gcs_uri:
                    next_updates["output_prefix"] = remote.dest.gcs_uri
                if remote.error and getattr(remote.error, "message", ""):
                    next_updates["error"] = str(remote.error.message)

                if job.get("status") != remote_state or next_updates.get("output_prefix") != job.get("output_prefix"):
                    self.update_job(pdf_file, **next_updates)
                    updated["synced"] += 1
            except Exception as exc:
                updated["errors"] += 1
                logger.warning(f"同步 Batch job 失败 {job.get('job_name')}: {exc}")
        return updated


class VertexOCRProcessor:
    PAGE_URI_PATTERN = re.compile(r"page_(\d+)\.png$", re.IGNORECASE)

    def __init__(self, config: Config, logger: logging.Logger, *, use_batch_api: bool):
        self.config = config
        self.logger = logger
        self.use_batch_api = use_batch_api

        client_kwargs = {
            "vertexai": True,
            "http_options": types.HttpOptions(api_version="v1"),
        }
        if config.vertex_api_key and not use_batch_api:
            client_kwargs["api_key"] = config.vertex_api_key
        else:
            client_kwargs["project"] = config.vertex_project
            client_kwargs["location"] = config.vertex_location
            if config.vertex_api_key and use_batch_api:
                self.logger.warning(
                    "Batch 模式已忽略 VERTEX_API_KEY，改用 ADC + project/location。"
                )

        self.client = genai.Client(**client_kwargs)
        self.storage_client = storage.Client(project=config.vertex_project)

    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        self.logger.info(f"转换 PDF 为图片: {pdf_path.name}")
        images = convert_from_path(pdf_path, dpi=self.config.dpi, fmt="PNG")
        self.logger.info(f"共 {len(images)} 页")
        return images

    def image_to_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def ocr_single_page(self, image: Image.Image, page_num: int) -> Tuple[int, str, Optional[str]]:
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=self.config.ocr_prompt),
                                types.Part.from_bytes(
                                    data=self.image_to_bytes(image),
                                    mime_type="image/png",
                                ),
                            ],
                        )
                    ],
                )

                if response.text:
                    return page_num, response.text.strip(), None
                return page_num, "", "Empty response"
            except Exception as exc:
                error = str(exc)
                self.logger.warning(f"页面 {page_num} 第 {attempt + 1} 次尝试失败: {error}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return page_num, "", error
        return page_num, "", "Max retries exceeded"

    def process_pdf_realtime(self, pdf_path: Path, status_manager: StatusManager) -> str:
        pdf_name = pdf_path.name
        images = self.pdf_to_images(pdf_path)

        with tqdm(total=len(images), desc=pdf_name, unit="页") as progress:
            for page_num, image in enumerate(images, 1):
                if status_manager.is_completed(pdf_name, page_num):
                    progress.update(1)
                    continue

                _, content, error = self.ocr_single_page(image, page_num)
                if error:
                    status_manager.mark_failed(pdf_name, page_num, error)
                    self.logger.error(f"页面 {page_num} 处理失败: {error}")
                else:
                    status_manager.mark_completed(pdf_name, page_num, content)
                progress.update(1)
                time.sleep(self.config.request_delay)

        return self._assemble_realtime_markdown(pdf_name, len(images), status_manager)

    def _assemble_realtime_markdown(
        self,
        pdf_name: str,
        total_pages: int,
        status_manager: StatusManager,
    ) -> str:
        full_content: List[str] = []
        for page_num in range(1, total_pages + 1):
            page_content = status_manager.get_page_content(pdf_name, page_num)
            if page_content is not None:
                full_content.append(f"<!-- Page {page_num} -->\n\n{page_content}")
                continue

            error = status_manager.get_page_error(pdf_name, page_num)
            if error:
                full_content.append(f"<!-- Page {page_num}: OCR Failed - {error} -->")
            else:
                full_content.append(f"<!-- Page {page_num}: No result -->")
        return "\n\n---\n\n".join(full_content)

    def wait_for_submission_slot(
        self,
        job_manager: BatchJobManager,
        *,
        allow_wait: bool,
    ) -> bool:
        limit = self.config.max_active_batch_jobs
        if limit is None or limit <= 0:
            return True

        while True:
            job_manager.sync_from_remote(self.client, self.logger)
            active_count = len(job_manager.get_active_jobs())
            if active_count < limit:
                return True

            if not allow_wait:
                self.logger.info(f"当前活跃 Batch Job 数 {active_count} 已达到上限 {limit}，停止继续提交。")
                return False

            self.logger.info(
                f"当前活跃 Batch Job 数 {active_count} 已达到上限 {limit}，"
                f"等待 {self.config.poll_interval} 秒后重试。"
            )
            time.sleep(self.config.poll_interval)

    def _upload_to_gcs(self, gcs_uri: str, data: bytes, content_type: str) -> None:
        bucket_name, blob_name = _split_gcs_uri(gcs_uri)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)

    def _build_batch_artifacts(
        self,
        pdf_path: Path,
    ) -> Tuple[str, str, int]:
        pdf_name = pdf_path.name
        safe_pdf_id = _safe_ascii_name(pdf_path.stem)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_root = _join_gcs_uri(
            self.config.gcs_bucket,
            self.config.gcs_prefix,
            "inputs",
            safe_pdf_id,
        )
        output_prefix = _join_gcs_uri(
            self.config.gcs_bucket,
            self.config.gcs_prefix,
            "outputs",
            safe_pdf_id,
            run_id,
        )

        images = self.pdf_to_images(pdf_path)
        requests: List[Dict] = []
        request_lines: List[str] = []

        self.logger.info(f"上传 {pdf_name} 的页面图片到 GCS")
        for page_num, image in enumerate(tqdm(images, desc=f"{pdf_name} 上传", unit="页"), 1):
            page_uri = f"{input_root}/pages/page_{page_num:04d}.png"
            self._upload_to_gcs(page_uri, self.image_to_bytes(image), "image/png")
            request = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": self.config.ocr_prompt},
                                {
                                    "fileData": {
                                        "fileUri": page_uri,
                                        "mimeType": "image/png",
                                    }
                                },
                            ],
                        }
                    ]
                }
            }
            requests.append(request)
            request_lines.append(json.dumps(request, ensure_ascii=False))

        local_request_path = self.config.batch_requests_dir / f"{safe_pdf_id}_requests.jsonl"
        local_request_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")

        input_uri = f"{input_root}/requests.jsonl"
        self._upload_to_gcs(
            input_uri,
            local_request_path.read_bytes(),
            "application/jsonl",
        )
        self.logger.info(f"已上传 Batch 请求文件: {input_uri}")
        return input_uri, output_prefix, len(requests)

    def submit_pdf_batch_job(
        self,
        pdf_path: Path,
        job_manager: BatchJobManager,
        *,
        allow_wait_for_slot: bool,
    ) -> Dict[str, str]:
        if not self.wait_for_submission_slot(job_manager, allow_wait=allow_wait_for_slot):
            raise RuntimeError("batch_slot_limit_reached")

        input_uri, output_prefix, page_count = self._build_batch_artifacts(pdf_path)
        display_name = f"{_safe_ascii_name(pdf_path.stem)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.logger.info(f"提交 Vertex Batch Job: {pdf_path.name}")
        try:
            batch_job = self.client.batches.create(
                model=self.config.model_name,
                src=input_uri,
                config=types.CreateBatchJobConfig(
                    display_name=display_name,
                    dest=output_prefix,
                ),
            )
        except Exception as exc:
            msg = str(exc)
            if "aiplatform.batchPredictionJobs.create" in msg:
                raise RuntimeError(
                    "缺少 Vertex Batch 创建权限。请使用 ADC 凭证并给当前账号/服务账号授予 "
                    "`aiplatform.batchPredictionJobs.create`（如 `roles/aiplatform.user`）后重试。"
                ) from exc
            raise

        job_state = _state_name(batch_job.state)
        job_data = {
            "job_name": batch_job.name or "",
            "status": job_state,
            "created_at": datetime.now().isoformat(),
            "completed_at": "",
            "input_uri": input_uri,
            "output_prefix": output_prefix,
            "page_count": str(page_count),
            "model_name": self.config.model_name,
            "error": "",
        }
        job_manager.upsert_job(pdf_path.name, job_data)
        self.logger.info(f"已创建 Batch Job: {job_data['job_name']} ({job_state})")
        return job_data

    def wait_for_job(self, pdf_file: str, job_manager: BatchJobManager) -> Dict[str, str]:
        job = job_manager.get_job(pdf_file)
        if not job:
            raise RuntimeError(f"找不到 {pdf_file} 的 Batch Job")

        while True:
            remote = self.client.batches.get(name=job["job_name"])
            state = _state_name(remote.state)
            updates: Dict[str, str] = {"status": state}
            if remote.dest and remote.dest.gcs_uri:
                updates["output_prefix"] = remote.dest.gcs_uri
            if remote.error and getattr(remote.error, "message", ""):
                updates["error"] = str(remote.error.message)

            job_manager.update_job(pdf_file, **updates)
            job = job_manager.get_job(pdf_file) or {}

            if state in BatchJobManager.TERMINAL_STATES:
                self.logger.info(f"{pdf_file} 的 Batch Job 已结束: {state}")
                return job

            self.logger.info(f"{pdf_file} 的 Batch Job 状态: {state}，{self.config.poll_interval} 秒后继续轮询")
            time.sleep(self.config.poll_interval)

    def _list_output_jsonl_uris(self, output_prefix: str) -> List[str]:
        bucket_name, prefix = _split_gcs_uri(output_prefix)
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix)
        uris = [
            f"gs://{bucket_name}/{blob.name}"
            for blob in blobs
            if blob.name.endswith(".jsonl")
        ]
        return sorted(uris)

    def _download_gcs_text(self, gcs_uri: str) -> str:
        bucket_name, blob_name = _split_gcs_uri(gcs_uri)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")

    def _extract_page_num_from_request(self, request: Dict) -> Optional[int]:
        for content in request.get("contents", []):
            for part in content.get("parts", []):
                file_data = part.get("fileData") or {}
                file_uri = file_data.get("fileUri", "")
                match = self.PAGE_URI_PATTERN.search(file_uri)
                if match:
                    return int(match.group(1))
        return None

    def collect_batch_results(self, pdf_file: str, job_manager: BatchJobManager) -> Tuple[str, Dict[str, int]]:
        job = job_manager.get_job(pdf_file)
        if not job:
            raise RuntimeError(f"找不到 {pdf_file} 的 Batch Job")

        output_prefix = job.get("output_prefix", "")
        if not output_prefix:
            raise RuntimeError(f"{pdf_file} 的 Batch Job 尚未记录输出目录")

        output_uris = self._list_output_jsonl_uris(output_prefix)
        if not output_uris:
            raise RuntimeError(f"{pdf_file} 的 Batch 输出目录为空: {output_prefix}")

        safe_pdf_id = _safe_ascii_name(Path(pdf_file).stem)
        local_dir = self.config.batch_results_dir / safe_pdf_id
        local_dir.mkdir(parents=True, exist_ok=True)

        page_results: Dict[int, str] = {}
        usage = {
            "prompt_tokens": 0,
            "candidates_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }

        for output_uri in output_uris:
            local_path = local_dir / Path(_split_gcs_uri(output_uri)[1]).name
            content = self._download_gcs_text(output_uri)
            local_path.write_text(content, encoding="utf-8")

            for line in content.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                page_num = self._extract_page_num_from_request(row.get("request", {}))
                if page_num is None:
                    self.logger.warning(f"无法从输出中解析页码: {output_uri}")
                    continue

                status = row.get("status", "")
                response = row.get("response", {})
                if status:
                    page_results[page_num] = f"<!-- Page {page_num}: OCR Failed - {status} -->"
                    continue

                page_results[page_num] = _extract_text_from_response(response) or f"<!-- Page {page_num}: Empty response -->"

                usage_meta = response.get("usageMetadata", {})
                usage["prompt_tokens"] += usage_meta.get("promptTokenCount", 0)
                usage["candidates_tokens"] += usage_meta.get("candidatesTokenCount", 0)
                usage["thinking_tokens"] += usage_meta.get("thoughtsTokenCount", 0)
                usage["total_tokens"] += usage_meta.get("totalTokenCount", 0)
                usage["request_count"] += 1

        total_pages = int(job.get("page_count", "0") or 0)
        chunks: List[str] = []
        for page_num in range(1, total_pages + 1):
            content = page_results.get(page_num, f"<!-- Page {page_num}: No result -->")
            chunks.append(f"<!-- Page {page_num} -->\n\n{content}")
        return "\n\n---\n\n".join(chunks), usage

    def save_markdown(self, pdf_path: Path, markdown_content: str, *, mode_label: str) -> Path:
        output_path = self.config.output_dir / f"{pdf_path.stem}.md"
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"# {pdf_path.stem}\n\n")
            handle.write(f"*OCR processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            handle.write(f"*Model: {self.config.model_name} ({mode_label})*\n\n")
            handle.write("---\n\n")
            handle.write(markdown_content)
        return output_path

    def save_batch_markdown(self, pdf_path: Path, job_manager: BatchJobManager) -> Path:
        markdown_content, usage = self.collect_batch_results(pdf_path.name, job_manager)
        output_path = self.save_markdown(pdf_path, markdown_content, mode_label="Vertex Batch")
        self.logger.info(f"已保存: {output_path}")
        self.logger.info(
            "Batch token 使用: requests=%s prompt=%s output=%s thinking=%s total=%s",
            usage["request_count"],
            usage["prompt_tokens"],
            usage["candidates_tokens"],
            usage["thinking_tokens"],
            usage["total_tokens"],
        )
        return output_path


def find_pdf_files(input_dir: Path) -> List[Path]:
    return sorted(list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF")))


def print_job_status(job_manager: BatchJobManager, logger: logging.Logger) -> None:
    stats = job_manager.get_stats()
    logger.info("=" * 60)
    logger.info("Vertex Batch Job 状态")
    logger.info(f"待处理: {stats['pending']}")
    logger.info(f"运行中: {stats['running']}")
    logger.info(f"成功: {stats['succeeded']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"总计: {stats['total']}")
    logger.info("=" * 60)

    for pdf_file, job in job_manager.all_jobs():
        logger.info(
            "%s: %s | %s",
            pdf_file,
            job.get("status", ""),
            job.get("job_name", ""),
        )


def cleanup_all_batch_jobs(
    job_manager: BatchJobManager,
    client: genai.Client,
    logger: logging.Logger,
    *,
    target_pdf_files: Optional[List[str]] = None,
) -> Dict[str, int]:
    if target_pdf_files:
        target_set = set(target_pdf_files)
        all_jobs = [
            (pdf_file, job)
            for pdf_file, job in job_manager.all_jobs()
            if pdf_file in target_set
        ]
    else:
        all_jobs = job_manager.all_jobs()

    stats = {"total": len(all_jobs), "cancelled": 0, "deleted": 0, "errors": 0}
    if not all_jobs:
        logger.info("没有可清理的 Batch Job 记录")
        return stats

    for _, job in all_jobs:
        job_name = job.get("job_name", "")
        status = job.get("status", "")
        if not job_name:
            continue
        if status in BatchJobManager.ACTIVE_STATES:
            try:
                client.batches.cancel(name=job_name)
                stats["cancelled"] += 1
                logger.info(f"已取消: {job_name}")
            except Exception as exc:
                stats["errors"] += 1
                logger.warning(f"取消失败 {job_name}: {exc}")

    if stats["cancelled"] > 0:
        time.sleep(2)

    for _, job in all_jobs:
        job_name = job.get("job_name", "")
        if not job_name:
            continue
        try:
            delete_job = client.batches.delete(name=job_name)
            for _ in range(30):
                if getattr(delete_job, "done", False):
                    break
                time.sleep(1)
            stats["deleted"] += 1
            logger.info(f"已删除: {job_name}")
        except Exception as exc:
            stats["errors"] += 1
            logger.warning(f"删除失败 {job_name}: {exc}")

    if stats["deleted"] == stats["total"]:
        if target_pdf_files:
            for pdf_file in target_pdf_files:
                job_manager.jobs.pop(pdf_file, None)
        else:
            job_manager.jobs = {}
        job_manager._save()

    logger.info(
        "清理完成: 总计 %s，取消 %s，删除 %s，错误 %s",
        stats["total"],
        stats["cancelled"],
        stats["deleted"],
        stats["errors"],
    )
    return stats


def cleanup_gcs_artifacts(
    job_manager: BatchJobManager,
    storage_client: storage.Client,
    logger: logging.Logger,
    *,
    target_pdf_files: Optional[List[str]] = None,
) -> Dict[str, int]:
    if target_pdf_files:
        target_set = set(target_pdf_files)
        target_jobs = [
            (pdf_file, job)
            for pdf_file, job in job_manager.all_jobs()
            if pdf_file in target_set
        ]
    else:
        target_jobs = job_manager.all_jobs()

    stats = {"jobs": len(target_jobs), "prefixes": 0, "deleted_objects": 0, "errors": 0}
    if not target_jobs:
        logger.info("没有可清理的 GCS 产物记录")
        return stats

    prefixes: set[Tuple[str, str]] = set()
    for _, job in target_jobs:
        input_uri = job.get("input_uri", "")
        if input_uri:
            try:
                bucket_name, blob_name = _split_gcs_uri(input_uri)
                input_prefix = str(Path(blob_name).parent).strip("/")
                if input_prefix:
                    prefixes.add((bucket_name, f"{input_prefix}/"))
            except Exception as exc:
                stats["errors"] += 1
                logger.warning(f"解析 input_uri 失败 {input_uri}: {exc}")

        output_prefix = job.get("output_prefix", "")
        if output_prefix:
            try:
                bucket_name, prefix = _split_gcs_uri(output_prefix)
                prefix = prefix.strip("/")
                if prefix:
                    prefixes.add((bucket_name, f"{prefix}/"))
            except Exception as exc:
                stats["errors"] += 1
                logger.warning(f"解析 output_prefix 失败 {output_prefix}: {exc}")

    stats["prefixes"] = len(prefixes)
    if not prefixes:
        logger.info("未找到可删除的 GCS 前缀")
        return stats

    for bucket_name, prefix in sorted(prefixes):
        logger.info(f"清理 GCS 前缀: gs://{bucket_name}/{prefix}")
        try:
            for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
                try:
                    blob.delete()
                    stats["deleted_objects"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    logger.warning(f"删除对象失败 gs://{bucket_name}/{blob.name}: {exc}")
        except Exception as exc:
            stats["errors"] += 1
            logger.warning(f"遍历前缀失败 gs://{bucket_name}/{prefix}: {exc}")

    cleaned_at = datetime.now().isoformat()
    for pdf_file, _ in target_jobs:
        if pdf_file in job_manager.jobs:
            job_manager.jobs[pdf_file]["gcs_cleaned_at"] = cleaned_at
    job_manager._save()

    logger.info(
        "GCS 清理完成: job %s 个，前缀 %s 个，删除对象 %s 个，错误 %s",
        stats["jobs"],
        stats["prefixes"],
        stats["deleted_objects"],
        stats["errors"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex OCR - 支持实时和 Batch 模式")
    parser.add_argument("--reset", action="store_true", help="重置本地状态文件")
    parser.add_argument("--file", type=str, help="只处理指定的 PDF 文件")
    parser.add_argument("--realtime", action="store_true", help="使用实时模式")
    parser.add_argument("--no-wait", action="store_true", help="Batch 模式: 提交后立即退出")
    parser.add_argument("--status", action="store_true", help="显示所有 Vertex Batch Job 状态")
    parser.add_argument("--download", action="store_true", help="下载并合并已完成的 Batch 结果")
    parser.add_argument("--cleanup-all-jobs", action="store_true", help="取消并删除本地记录中的所有 Batch Job")
    parser.add_argument("--cleanup-gcs-artifacts", action="store_true", help="删除本地记录对应的 GCS 输入/输出产物")
    parser.add_argument("--max-active-batch-jobs", type=int, help="限制同时活跃的 Batch Job 数")
    args = parser.parse_args()

    config = Config.load()
    if args.max_active_batch_jobs:
        config.max_active_batch_jobs = args.max_active_batch_jobs

    use_batch_api = not args.realtime
    logger = setup_logging(config.logs_dir)
    if not config.validate(use_batch_api=use_batch_api):
        sys.exit(1)

    if args.reset:
        for path in [config.status_file, config.batch_status_file]:
            if path.exists():
                path.unlink()
                logger.info(f"已删除状态文件: {path}")

    if args.file:
        pdf_files = [config.input_dir / args.file]
        if not (args.status or args.download or args.cleanup_all_jobs or args.cleanup_gcs_artifacts) and not pdf_files[0].exists():
            logger.error(f"文件不存在: {args.file}")
            sys.exit(1)
    else:
        pdf_files = find_pdf_files(config.input_dir)

    if not pdf_files and not args.status and not args.download and not args.cleanup_all_jobs and not args.cleanup_gcs_artifacts:
        logger.warning(f"未找到 PDF 文件，请将 PDF 放入 {config.input_dir}")
        sys.exit(0)

    processor = VertexOCRProcessor(config, logger, use_batch_api=use_batch_api)

    if use_batch_api:
        job_manager = BatchJobManager(config.batch_status_file)

        if args.status:
            logger.info("同步远程状态...")
            job_manager.sync_from_remote(processor.client, logger)
            print_job_status(job_manager, logger)
            return

        if args.cleanup_all_jobs:
            cleanup_targets = [pdf_path.name for pdf_path in pdf_files] if args.file else None
            cleanup_all_batch_jobs(
                job_manager,
                processor.client,
                logger,
                target_pdf_files=cleanup_targets,
            )
            return

        if args.cleanup_gcs_artifacts:
            cleanup_targets = [pdf_path.name for pdf_path in pdf_files] if args.file else None
            cleanup_gcs_artifacts(
                job_manager,
                processor.storage_client,
                logger,
                target_pdf_files=cleanup_targets,
            )
            return

        logger.info("=" * 60)
        logger.info("Vertex OCR 启动 [Batch]")
        logger.info(f"模型: {config.model_name}")
        logger.info(f"项目: {config.vertex_project}")
        logger.info(f"区域: {config.vertex_location}")
        logger.info("Vertex 认证: ADC / 默认凭证 (Batch 模式)")
        logger.info(f"GCS Bucket: {config.gcs_bucket}")
        logger.info(f"GCS Prefix: {config.gcs_prefix}")
        logger.info("=" * 60)

        job_manager.sync_from_remote(processor.client, logger)

        if args.download:
            # 下载模式允许 input/ 为空；此时从状态文件恢复待处理 PDF 列表。
            if not pdf_files:
                pdf_files = sorted(config.input_dir / pdf_name for pdf_name, _ in job_manager.all_jobs())
                if not pdf_files:
                    logger.warning("下载模式下未找到 Batch Job 记录，无法下载结果")
                    return

            for pdf_path in pdf_files:
                job = job_manager.get_job(pdf_path.name)
                if not job:
                    logger.warning(f"{pdf_path.name}: 没有 Batch Job 记录")
                    continue
                if job.get("status") != "JOB_STATE_SUCCEEDED":
                    logger.warning(f"{pdf_path.name}: 当前状态 {job.get('status')}，暂不下载")
                    continue
                processor.save_batch_markdown(pdf_path, job_manager)
            print_job_status(job_manager, logger)
            return

        saved_outputs: set[str] = set()
        failed_outputs: set[str] = set()

        # 阶段1：先提交/复用全部任务（等待模式下不会“一个 PDF 提交后立即等完”）
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name
            job = job_manager.get_job(pdf_name)

            if job and job.get("status") in BatchJobManager.ACTIVE_STATES:
                logger.info(f"{pdf_name}: 已有活跃 Batch Job，复用并继续处理下一个 PDF")
                continue

            if job and job.get("status") == "JOB_STATE_SUCCEEDED":
                logger.info(f"{pdf_name}: Batch 已完成，直接合并结果")
                processor.save_batch_markdown(pdf_path, job_manager)
                saved_outputs.add(pdf_name)
                continue

            try:
                processor.submit_pdf_batch_job(
                    pdf_path,
                    job_manager,
                    allow_wait_for_slot=not args.no_wait,
                )
            except RuntimeError as exc:
                if str(exc) == "batch_slot_limit_reached":
                    logger.info("已达到活跃 Batch Job 上限，本轮停止继续提交")
                    break
                logger.error(str(exc))
                return

        if args.no_wait:
            print_job_status(job_manager, logger)
            return

        # 阶段2：统一轮询并回收结果
        target_pdf_names = [pdf_path.name for pdf_path in pdf_files]
        pdf_by_name = {pdf_path.name: pdf_path for pdf_path in pdf_files}

        while True:
            job_manager.sync_from_remote(processor.client, logger)

            unfinished: List[str] = []
            active_count = 0

            for pdf_name in target_pdf_names:
                job = job_manager.get_job(pdf_name)
                if not job:
                    unfinished.append(pdf_name)
                    continue

                status = job.get("status", "")
                if status in BatchJobManager.ACTIVE_STATES:
                    unfinished.append(pdf_name)
                    active_count += 1
                    continue

                if status == "JOB_STATE_SUCCEEDED":
                    if pdf_name not in saved_outputs:
                        try:
                            processor.save_batch_markdown(pdf_by_name[pdf_name], job_manager)
                            saved_outputs.add(pdf_name)
                        except Exception as exc:
                            logger.error(f"{pdf_name}: 合并 Batch 结果失败: {exc}")
                    continue

                if status in BatchJobManager.TERMINAL_STATES:
                    if pdf_name not in failed_outputs:
                        logger.error(
                            "%s 的 Batch Job 结束但未成功: %s %s",
                            pdf_name,
                            status,
                            job.get("error", ""),
                        )
                        failed_outputs.add(pdf_name)
                    continue

                unfinished.append(pdf_name)

            if not unfinished:
                break

            logger.info(
                "统一轮询中: 剩余未完成 PDF %s 个，活跃 Batch Job %s 个，%s 秒后继续。",
                len(unfinished),
                active_count,
                config.poll_interval,
            )
            time.sleep(config.poll_interval)

        print_job_status(job_manager, logger)
        return

    status_manager = StatusManager(config.status_file, config.page_cache_dir)

    logger.info("=" * 60)
    logger.info("Vertex OCR 启动 [实时]")
    logger.info(f"模型: {config.model_name}")
    logger.info(f"项目: {config.vertex_project}")
    logger.info(f"区域: {config.vertex_location}")
    logger.info(f"Vertex 认证: {'API key' if config.vertex_api_key else 'ADC / 默认凭证'}")
    logger.info("=" * 60)

    for pdf_path in pdf_files:
        try:
            markdown_content = processor.process_pdf_realtime(pdf_path, status_manager)
            output_path = processor.save_markdown(pdf_path, markdown_content, mode_label="Vertex Realtime")
            logger.info(f"已保存: {output_path}")
        except Exception as exc:
            logger.error(f"处理 {pdf_path.name} 时出错: {exc}")

    stats = status_manager.get_stats()
    logger.info("=" * 60)
    logger.info("处理完成")
    logger.info(f"成功: {stats['completed']} 页")
    logger.info(f"失败: {stats['failed']} 页")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
