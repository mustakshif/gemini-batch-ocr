#!/usr/bin/env python3
"""
Gemini Batch OCR Script
=======================
批量处理扫描版 PDF，默认使用 Gemini 3.1 Flash-Lite Preview 进行 OCR，输出 Markdown 格式。

功能特性：
- 支持阿拉伯语等复杂语种（RTL脚本）
- 断点续传：中断后可继续处理
- 失败重试：自动重试失败的页面
- 详细日志：记录处理进度和错误
- 并发控制：避免API限流

使用方法：
1. 复制 config.example.env 为 .env 并填入API密钥
2. 将PDF文件放入 input/ 目录
3. 运行: python batch_ocr.py
4. 输出在 output/ 目录

作者: AI Assistant
日期: 2025-01
"""

import os
import sys
import csv
import time
import json
import base64
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

try:
    from google import genai
    from google.genai import types
    from pdf2image import convert_from_path
    from PIL import Image
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# 配置
# ============================================================================

class Config:
    """配置管理"""
    
    def __init__(self):
        load_dotenv()
        
        # API配置
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
        
        # 路径配置
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.logs_dir = self.base_dir / "logs"
        
        # 处理配置（实时模式）
        self.dpi = int(os.getenv("PDF_DPI", "200"))  # PDF转图片DPI
        self.max_workers = int(os.getenv("MAX_WORKERS", "3"))  # 并发数
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))  # 重试次数
        self.retry_delay = float(os.getenv("RETRY_DELAY", "5"))  # 重试延迟(秒)
        self.request_delay = float(os.getenv("REQUEST_DELAY", "1"))  # 请求间隔(秒)
        
        # Batch API 配置
        self.use_batch_api = True  # 默认使用 Batch API
        self.batch_size = int(os.getenv("BATCH_SIZE", "50"))  # 每批最大页数
        self.poll_interval = int(os.getenv("POLL_INTERVAL", "30"))  # 轮询间隔(秒)
        self.max_active_batch_jobs = _optional_int_env("MAX_ACTIVE_BATCH_JOBS")  # 同时挂起/运行的最大 Batch Job 数，默认不启用
        self.wait_for_completion = True  # 是否等待完成
        
        # Batch API 目录
        self.batch_requests_dir = self.logs_dir / "batch_requests"
        self.batch_results_dir = self.logs_dir / "batch_results"
        self.batch_status_file = self.logs_dir / "batch_jobs.json"
        
        # 语言配置
        self.primary_language = os.getenv("PRIMARY_LANGUAGE", "Arabic")
        self.ocr_prompt = os.getenv("OCR_PROMPT", self._default_prompt())
        self.ocr_prompt_after_meta_pages = os.getenv("OCR_PROMPT_AFTER_META_PAGES", self.ocr_prompt)
        self.ocr_meta_page_limit = max(0, int(os.getenv("OCR_META_PAGE_LIMIT", "10")))
        
        # 状态文件（实时模式）
        self.status_file = self.logs_dir / "processing_status.csv"
        
    def _default_prompt(self) -> str:
        return f"""You are an expert OCR system specialized in {self.primary_language} document processing.

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

    def validate(self) -> bool:
        if not self.gemini_api_key:
            logging.error("GEMINI_API_KEY 未设置")
            return False
        
        dirs = [self.input_dir, self.output_dir, self.logs_dir]
        if self.use_batch_api:
            dirs.extend([self.batch_requests_dir, self.batch_results_dir])
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return True

    def prompt_for_page(self, page_num: int) -> str:
        if self.ocr_meta_page_limit > 0 and page_num > self.ocr_meta_page_limit:
            return self.ocr_prompt_after_meta_pages
        return self.ocr_prompt


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(logs_dir: Path) -> logging.Logger:
    """配置日志"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"batch_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# 状态管理（支持断点续传）
# ============================================================================

class StatusManager:
    """处理状态管理器"""
    
    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.status: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """加载状态文件"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row['pdf_file']}:{row['page_num']}"
                    self.status[key] = row
    
    def _save(self):
        """保存状态文件"""
        if not self.status:
            return
            
        fieldnames = ['pdf_file', 'page_num', 'status', 'timestamp', 'error']
        with open(self.status_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.status.values())
    
    def is_completed(self, pdf_file: str, page_num: int) -> bool:
        """检查页面是否已完成"""
        key = f"{pdf_file}:{page_num}"
        return self.status.get(key, {}).get('status') == 'completed'
    
    def mark_completed(self, pdf_file: str, page_num: int):
        """标记页面完成"""
        key = f"{pdf_file}:{page_num}"
        self.status[key] = {
            'pdf_file': pdf_file,
            'page_num': page_num,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'error': ''
        }
        self._save()
    
    def mark_failed(self, pdf_file: str, page_num: int, error: str):
        """标记页面失败"""
        key = f"{pdf_file}:{page_num}"
        self.status[key] = {
            'pdf_file': pdf_file,
            'page_num': page_num,
            'status': 'failed',
            'timestamp': datetime.now().isoformat(),
            'error': error[:200]  # 截断错误信息
        }
        self._save()
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = {'completed': 0, 'failed': 0, 'total': len(self.status)}
        for item in self.status.values():
            if item['status'] == 'completed':
                stats['completed'] += 1
            elif item['status'] == 'failed':
                stats['failed'] += 1
        return stats


# ============================================================================
# Batch Job 状态管理
# ============================================================================

class BatchJobManager:
    """Batch Job 状态管理器"""
    
    COMPLETED_STATES = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    ACTIVE_STATES = {'pending', 'JOB_STATE_PENDING', 'JOB_STATE_RUNNING'}
    
    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.jobs: Dict[str, List[Dict]] = {}
        self._load()
    
    def _load(self):
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.jobs = json.load(f)
    
    def _save(self):
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.jobs, f, ensure_ascii=False, indent=2)
    
    def add_job(self, pdf_file: str, batch_index: int, job_name: str, 
                page_start: int, page_end: int, input_file: str):
        if pdf_file not in self.jobs:
            self.jobs[pdf_file] = []
        
        job_entry = {
            'batch_index': batch_index,
            'job_name': job_name,
            'page_start': page_start,
            'page_end': page_end,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'completed_at': None,
            'input_file': input_file,
            'result_file': None,
            'error': None
        }
        
        existing = self._find_job(pdf_file, batch_index)
        if existing:
            idx = self.jobs[pdf_file].index(existing)
            self.jobs[pdf_file][idx] = job_entry
        else:
            self.jobs[pdf_file].append(job_entry)
        
        self._save()
    
    def _find_job(self, pdf_file: str, batch_index: int) -> Optional[Dict]:
        if pdf_file not in self.jobs:
            return None
        for job in self.jobs[pdf_file]:
            if job['batch_index'] == batch_index:
                return job
        return None
    
    def update_job_status(self, pdf_file: str, batch_index: int, status: str,
                          result_file: Optional[str] = None, error: Optional[str] = None):
        job = self._find_job(pdf_file, batch_index)
        if job:
            job['status'] = status
            if status in self.COMPLETED_STATES:
                job['completed_at'] = datetime.now().isoformat()
            if result_file:
                job['result_file'] = result_file
            if error:
                job['error'] = error[:500]
            self._save()
    
    def update_job_by_name(self, job_name: str, status: str, 
                           result_file: Optional[str] = None, error: Optional[str] = None):
        for pdf_file, jobs in self.jobs.items():
            for job in jobs:
                if job['job_name'] == job_name:
                    job['status'] = status
                    if status in self.COMPLETED_STATES:
                        job['completed_at'] = datetime.now().isoformat()
                    if result_file:
                        job['result_file'] = result_file
                    if error:
                        job['error'] = error[:500]
                    self._save()
                    return
    
    def get_pending_jobs(self, pdf_file: Optional[str] = None) -> List[Dict]:
        pending = []
        target_jobs = self.jobs.get(pdf_file, []) if pdf_file else \
                      [job for jobs in self.jobs.values() for job in jobs]
        for job in target_jobs:
            if job['status'] in self.ACTIVE_STATES:
                pending.append(job)
        return pending

    def get_active_jobs(self) -> List[Dict]:
        return [
            job
            for jobs in self.jobs.values()
            for job in jobs
            if job['status'] in self.ACTIVE_STATES
        ]

    def get_all_jobs(self) -> List[Tuple[str, Dict]]:
        flattened: List[Tuple[str, Dict]] = []
        for pdf_file, jobs in self.jobs.items():
            for job in jobs:
                flattened.append((pdf_file, job))
        return flattened
    
    def get_succeeded_jobs(self, pdf_file: Optional[str] = None) -> List[Dict]:
        succeeded = []
        target_jobs = self.jobs.get(pdf_file, []) if pdf_file else \
                      [job for jobs in self.jobs.values() for job in jobs]
        for job in target_jobs:
            if job['status'] == 'JOB_STATE_SUCCEEDED':
                succeeded.append(job)
        return succeeded
    
    def is_batch_completed(self, pdf_file: str, batch_index: int) -> bool:
        job = self._find_job(pdf_file, batch_index)
        return job is not None and job['status'] == 'JOB_STATE_SUCCEEDED'
    
    def get_all_jobs_for_pdf(self, pdf_file: str) -> List[Dict]:
        return self.jobs.get(pdf_file, [])
    
    def get_stats(self, pdf_file: Optional[str] = None) -> Dict[str, int]:
        stats = {'pending': 0, 'running': 0, 'succeeded': 0, 'failed': 0, 'total': 0}
        target_jobs = self.jobs.get(pdf_file, []) if pdf_file else \
                      [job for jobs in self.jobs.values() for job in jobs]
        
        for job in target_jobs:
            stats['total'] += 1
            status = job['status']
            if status in ('pending', 'JOB_STATE_PENDING'):
                stats['pending'] += 1
            elif status == 'JOB_STATE_RUNNING':
                stats['running'] += 1
            elif status == 'JOB_STATE_SUCCEEDED':
                stats['succeeded'] += 1
            elif status in ('JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
                stats['failed'] += 1
        
        return stats
    
    def get_all_job_names(self) -> List[str]:
        names = []
        for jobs in self.jobs.values():
            for job in jobs:
                if job['job_name']:
                    names.append(job['job_name'])
        return names
    
    def sync_from_remote(self, client) -> Dict[str, int]:
        """从远程 API 同步所有 job 状态，返回更新统计"""
        updated = {'synced': 0, 'errors': 0}
        
        for pdf_file, job_list in self.jobs.items():
            for job in job_list:
                if job['status'] in self.COMPLETED_STATES:
                    continue  # 已完成的不需要再查
                
                try:
                    batch_job = client.batches.get(name=job['job_name'])
                    state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                    
                    if job['status'] != state:
                        job['status'] = state
                        if state in self.COMPLETED_STATES:
                            job['completed_at'] = datetime.now().isoformat()
                        if batch_job.dest and batch_job.dest.file_name:
                            job['result_file'] = batch_job.dest.file_name
                        updated['synced'] += 1
                        
                except Exception as e:
                    updated['errors'] += 1
                    logging.warning(f"同步 {job['job_name']} 失败: {e}")
        
        if updated['synced'] > 0:
            self._save()
        
        return updated


# ============================================================================
# OCR处理器
# ============================================================================

class GeminiOCRProcessor:
    """Gemini OCR处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.queue_limit_reached = False
        
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """将PDF转换为图片列表"""
        self.logger.info(f"转换PDF为图片: {pdf_path.name}")
        
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.config.dpi,
                fmt='PNG'
            )
            self.logger.info(f"共 {len(images)} 页")
            return images
        except Exception as e:
            self.logger.error(f"PDF转换失败: {e}")
            raise
    
    def image_to_bytes(self, image: Image.Image) -> bytes:
        """将PIL Image转换为bytes"""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def ocr_single_page(self, image: Image.Image, page_num: int) -> Tuple[int, str, Optional[str]]:
        """
        OCR单页
        返回: (page_num, markdown_content, error_message)
        """
        for attempt in range(self.config.max_retries):
            try:
                image_bytes = self.image_to_bytes(image)
                prompt = self.config.prompt_for_page(page_num)
                
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=prompt),
                                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                            ]
                        )
                    ]
                )
                
                if response.text:
                    return (page_num, response.text.strip(), None)
                else:
                    return (page_num, "", "Empty response")
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"页面 {page_num} 第 {attempt+1} 次尝试失败: {error_msg}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return (page_num, "", error_msg)
        
        return (page_num, "", "Max retries exceeded")
    
    def process_pdf(self, pdf_path: Path, status_manager: StatusManager) -> str:
        """
        处理单个PDF文件
        返回: 完整的Markdown内容
        """
        pdf_name = pdf_path.name
        self.logger.info(f"开始处理: {pdf_name}")
        
        # 转换PDF为图片
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)
        
        # 准备结果容器
        results: Dict[int, str] = {}
        
        # 处理每一页
        with tqdm(total=total_pages, desc=pdf_name, unit="页") as pbar:
            for page_num, image in enumerate(images, 1):
                # 检查是否已完成（断点续传）
                if status_manager.is_completed(pdf_name, page_num):
                    self.logger.debug(f"页面 {page_num} 已完成，跳过")
                    pbar.update(1)
                    continue
                
                # OCR处理
                _, content, error = self.ocr_single_page(image, page_num)
                
                if error:
                    status_manager.mark_failed(pdf_name, page_num, error)
                    self.logger.error(f"页面 {page_num} 处理失败: {error}")
                    results[page_num] = f"\n\n<!-- Page {page_num}: OCR Failed - {error} -->\n\n"
                else:
                    status_manager.mark_completed(pdf_name, page_num)
                    results[page_num] = content
                
                pbar.update(1)
                
                # 请求间隔（避免限流）
                time.sleep(self.config.request_delay)
        
        # 如果有断点续传，需要从输出文件读取已完成的内容
        output_path = self.config.output_dir / f"{pdf_path.stem}.md"
        if output_path.exists():
            # 简单处理：重新合并所有页面
            pass
        
        # 合并所有页面
        full_content = []
        for page_num in range(1, total_pages + 1):
            if page_num in results:
                full_content.append(f"<!-- Page {page_num} -->\n\n{results[page_num]}")
            else:
                # 从之前的处理中恢复（如果需要更完善的断点续传，这里需要扩展）
                full_content.append(f"\n\n<!-- Page {page_num}: Content from previous run -->\n\n")
        
        return "\n\n---\n\n".join(full_content)
    
    # ========================================================================
    # Batch API 方法
    # ========================================================================
    
    def _safe_ascii_name(self, name: str) -> str:
        import hashlib
        if name.isascii():
            return name
        hash_suffix = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        return f"pdf_{hash_suffix}"
    
    def image_to_base64(self, image: Image.Image) -> str:
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def prepare_batch_request(self, image: Image.Image, safe_pdf_id: str, page_num: int) -> Dict:
        image_b64 = self.image_to_base64(image)
        prompt = self.config.prompt_for_page(page_num)
        return {
            "key": f"{safe_pdf_id}_page_{page_num:04d}",
            "request": {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}}
                    ],
                    "role": "user"
                }]
            }
        }
    
    def create_jsonl_file(self, requests: List[Dict], output_path: Path) -> Path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        self.logger.info(f"创建 JSONL 文件: {output_path} ({len(requests)} 个请求)")
        return output_path
    
    def upload_jsonl_file(self, file_path: Path) -> str:
        self.logger.info(f"上传文件到 File API: {file_path.name}")
        safe_display_name = file_path.stem.encode('ascii', 'replace').decode('ascii')
        uploaded = self.client.files.upload(
            file=str(file_path),
            config=types.UploadFileConfig(
                display_name=safe_display_name,
                mime_type='application/jsonl'
            )
        )
        if not uploaded.name:
            raise RuntimeError("File upload failed: no name returned")
        self.logger.info(f"上传成功: {uploaded.name}")
        return uploaded.name
    
    def submit_batch_job(self, file_name: str, display_name: str) -> str:
        self.logger.info(f"提交 Batch Job: {display_name}")
        safe_display_name = display_name.encode('ascii', 'replace').decode('ascii')
        batch_job = self.client.batches.create(
            model=self.config.model_name,
            src=file_name,
            config={'display_name': safe_display_name}
        )
        if not batch_job.name:
            raise RuntimeError("Batch job creation failed: no name returned")
        self.logger.info(f"Job 已创建: {batch_job.name}")
        return batch_job.name

    def wait_for_submission_slot(self, job_manager: BatchJobManager, allow_wait: bool) -> bool:
        """在提交新 job 前控制活跃队列大小。"""
        while True:
            sync_result = job_manager.sync_from_remote(self.client)
            if sync_result['errors'] > 0:
                self.logger.warning(f"同步远程状态时出现 {sync_result['errors']} 个错误")
            active_count = len(job_manager.get_active_jobs())
            limit = self.config.max_active_batch_jobs

            if limit is None or limit <= 0:
                return True

            if active_count < limit:
                return True

            if not allow_wait:
                self.logger.info(
                    f"当前活跃 Batch Job 数 {active_count} 已达到上限 {limit}，停止继续提交。"
                )
                if not self.config.wait_for_completion:
                    self.logger.info("稍后重新执行 --no-wait，可继续补位提交剩余文件。")
                self.queue_limit_reached = True
                return False

            self.logger.info(
                f"当前活跃 Batch Job 数 {active_count} 已达到上限 {limit}，"
                f"等待 {self.config.poll_interval} 秒后重试提交。"
            )
            time.sleep(self.config.poll_interval)
    
    def poll_job_status(self, job_name: str) -> Dict:
        for attempt in range(self.config.max_retries):
            try:
                batch_job = self.client.batches.get(name=job_name)
                state_name = ''
                if batch_job.state:
                    state_name = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                return {
                    'name': batch_job.name,
                    'state': state_name,
                    'dest': batch_job.dest
                }
            except Exception as e:
                self.logger.warning(f"轮询 {job_name} 失败 (尝试 {attempt+1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise
        return {'name': job_name, 'state': 'POLL_ERROR', 'dest': None}
    
    def poll_jobs_until_complete(self, job_names: List[str]) -> Dict[str, Dict]:
        results = {}
        pending = set(job_names)
        
        while pending:
            self.logger.info(f"轮询中... 剩余 {len(pending)} 个 Job")
            
            for job_name in list(pending):
                status = self.poll_job_status(job_name)
                state = status['state']
                
                if state in BatchJobManager.COMPLETED_STATES:
                    results[job_name] = status
                    pending.remove(job_name)
                    self.logger.info(f"Job {job_name} 完成: {state}")
            
            if pending:
                self.logger.info(f"等待 {self.config.poll_interval} 秒...")
                time.sleep(self.config.poll_interval)
        
        return results
    
    def download_job_results(self, dest) -> str:
        if dest and dest.file_name:
            self.logger.info(f"下载结果文件: {dest.file_name}")
            content = self.client.files.download(file=dest.file_name)
            return content.decode('utf-8')
        elif dest and dest.inlined_responses:
            results = []
            for resp in dest.inlined_responses:
                if resp.response:
                    results.append(json.dumps({
                        'response': {'text': resp.response.text if hasattr(resp.response, 'text') else ''}
                    }))
            return '\n'.join(results)
        return ''
    
    def parse_batch_results(self, jsonl_content: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Parse batch results and extract token usage.
        
        Returns:
            Tuple of (results dict, usage stats dict)
            - results: {key: text_content}
            - usage: {prompt_tokens, candidates_tokens, thinking_tokens, total_tokens, request_count}
        """
        results = {}
        usage = {'prompt_tokens': 0, 'candidates_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0, 'request_count': 0}
        
        for line in jsonl_content.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                key = data.get('key', '')
                
                if 'response' in data and data['response']:
                    response = data['response']
                    candidates = response.get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts:
                            text = parts[0].get('text', '')
                            results[key] = text
                    
                    usage_meta = response.get('usageMetadata', {})
                    usage['prompt_tokens'] += usage_meta.get('promptTokenCount', 0)
                    usage['candidates_tokens'] += usage_meta.get('candidatesTokenCount', 0)
                    usage['thinking_tokens'] += usage_meta.get('thoughtsTokenCount', 0)
                    usage['total_tokens'] += usage_meta.get('totalTokenCount', 0)
                    usage['request_count'] += 1
                    
                elif 'error' in data:
                    results[key] = f"<!-- OCR Error: {data['error']} -->"
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"解析结果行失败: {e}")
        
        return results, usage
    
    def submit_pdf_batch_jobs(
        self,
        pdf_path: Path,
        job_manager: BatchJobManager,
        *,
        allow_wait_for_slot: bool,
    ) -> Tuple[int, bool]:
        pdf_name = pdf_path.name
        safe_pdf_id = self._safe_ascii_name(pdf_path.stem)
        self.queue_limit_reached = False
        self.logger.info(f"[Batch模式] 开始处理: {pdf_name}")
        
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)
        
        batch_size = self.config.batch_size
        num_batches = (total_pages + batch_size - 1) // batch_size
        existing_jobs = job_manager.get_all_jobs_for_pdf(pdf_name)
        existing_active = sum(1 for job in existing_jobs if job['status'] in BatchJobManager.ACTIVE_STATES)
        existing_done = sum(1 for job in existing_jobs if job['status'] == 'JOB_STATE_SUCCEEDED')
        
        self.logger.info(f"共 {total_pages} 页，分 {num_batches} 批处理 (每批 {batch_size} 页)")
        if existing_jobs:
            self.logger.info(
                f"已有 Batch 记录: 已完成 {existing_done} 批，活跃 {existing_active} 批，"
                f"待补提交 {max(0, num_batches - len(existing_jobs))} 批"
            )
        
        submitted_jobs = 0
        fully_submitted = True
        
        for batch_idx in range(num_batches):
            existing_job = job_manager._find_job(pdf_name, batch_idx)
            if existing_job:
                status = existing_job['status']
                if status == 'JOB_STATE_SUCCEEDED':
                    self.logger.info(f"批次 {batch_idx + 1}/{num_batches} 已完成，跳过")
                    continue
                if status in BatchJobManager.ACTIVE_STATES:
                    self.logger.info(
                        f"批次 {batch_idx + 1}/{num_batches} 已提交且仍在排队/运行 "
                        f"({status})，跳过重复提交"
                    )
                    continue

            if not self.wait_for_submission_slot(job_manager, allow_wait=allow_wait_for_slot):
                fully_submitted = False
                remaining_batches = num_batches - batch_idx
                self.logger.info(
                    f"{pdf_name}: 当前提交窗口已满，暂停补交；"
                    f"剩余未提交批次 {remaining_batches} 个。"
                )
                break
            
            page_start = batch_idx * batch_size + 1
            page_end = min((batch_idx + 1) * batch_size, total_pages)
            
            self.logger.info(f"处理批次 {batch_idx + 1}/{num_batches} (页 {page_start}-{page_end})")
            
            batch_requests = []
            for page_num in range(page_start, page_end + 1):
                image = images[page_num - 1]
                req = self.prepare_batch_request(image, safe_pdf_id, page_num)
                batch_requests.append(req)
            
            jsonl_path = self.config.batch_requests_dir / f"{safe_pdf_id}_batch_{batch_idx:03d}.jsonl"
            self.create_jsonl_file(batch_requests, jsonl_path)
            
            uploaded_file = self.upload_jsonl_file(jsonl_path)
            
            display_name = f"{safe_pdf_id}_batch_{batch_idx:03d}"
            job_name = self.submit_batch_job(uploaded_file, display_name)
            
            job_manager.add_job(pdf_name, batch_idx, job_name, page_start, page_end, uploaded_file)
            submitted_jobs += 1

        submitted_total = len(job_manager.get_all_jobs_for_pdf(pdf_name))
        remaining_unsubmitted = max(0, num_batches - submitted_total)
        self.logger.info(
            f"{pdf_name}: 本轮新提交 {submitted_jobs} 批；"
            f"累计已提交 {submitted_total}/{num_batches} 批；"
            f"剩余未提交 {remaining_unsubmitted} 批。"
        )

        if fully_submitted:
            self.logger.info(f"{pdf_name}: 所有批次均已提交，后续仅等待完成。")

        return submitted_jobs, fully_submitted

    def process_pdf_batch(self, pdf_path: Path, job_manager: BatchJobManager) -> Optional[Tuple[str, Dict[str, int]]]:
        _, _ = self.submit_pdf_batch_jobs(
            pdf_path,
            job_manager,
            allow_wait_for_slot=self.config.wait_for_completion,
        )

        if not self.config.wait_for_completion:
            self.logger.info("提交完成，--no-wait 模式，退出等待")
            self.logger.info(f"稍后使用 --download 获取结果")
            return None
        
        submitted_jobs = [j['job_name'] for j in job_manager.get_pending_jobs(pdf_name)]

        if submitted_jobs:
            self.logger.info(f"等待 {len(submitted_jobs)} 个 Job 完成...")
            completed = self.poll_jobs_until_complete(submitted_jobs)
            
            for job_name, status in completed.items():
                result_file = None
                if status['dest'] and status['dest'].file_name:
                    result_file = status['dest'].file_name
                job_manager.update_job_by_name(job_name, status['state'], result_file)
        
        return self.collect_batch_results(pdf_path, job_manager, total_pages)
    
    def collect_batch_results(self, pdf_path: Path, job_manager: BatchJobManager, 
                              total_pages: int) -> Tuple[str, Dict[str, int]]:
        """Collect and merge batch results with token usage statistics."""
        pdf_name = pdf_path.name
        safe_pdf_id = self._safe_ascii_name(pdf_path.stem)
        all_results: Dict[int, str] = {}
        total_usage = {'prompt_tokens': 0, 'candidates_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0, 'request_count': 0}
        
        for job in job_manager.get_all_jobs_for_pdf(pdf_name):
            if job['status'] != 'JOB_STATE_SUCCEEDED':
                for page in range(job['page_start'], job['page_end'] + 1):
                    all_results[page] = f"<!-- Page {page}: Batch Job Failed -->"
                continue
            
            result_path = self.config.batch_results_dir / f"{safe_pdf_id}_batch_{job['batch_index']:03d}_result.jsonl"
            
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8') as f:
                    jsonl_content = f.read()
            else:
                dest = self.client.batches.get(name=job['job_name']).dest
                jsonl_content = self.download_job_results(dest)
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(jsonl_content)
            
            parsed, usage = self.parse_batch_results(jsonl_content)
            
            for key, value in usage.items():
                total_usage[key] += value
            
            for key, content in parsed.items():
                try:
                    page_num = int(key.split('_page_')[1])
                    all_results[page_num] = content
                except (IndexError, ValueError):
                    self.logger.warning(f"无法解析 key: {key}")
        
        full_content = []
        for page_num in range(1, total_pages + 1):
            content = all_results.get(page_num, f"<!-- Page {page_num}: No result -->")
            full_content.append(f"<!-- Page {page_num} -->\n\n{content}")
        
        return "\n\n---\n\n".join(full_content), total_usage


# ============================================================================
# 主流程
# ============================================================================

def find_pdf_files(input_dir: Path) -> List[Path]:
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    return sorted(pdf_files)


def print_job_status(job_manager: BatchJobManager, logger: logging.Logger):
    stats = job_manager.get_stats()
    logger.info("=" * 60)
    logger.info("Batch Job 状态")
    logger.info(f"待处理: {stats['pending']}")
    logger.info(f"运行中: {stats['running']}")
    logger.info(f"成功: {stats['succeeded']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"总计: {stats['total']}")
    logger.info("=" * 60)
    
    for pdf_file, jobs in job_manager.jobs.items():
        logger.info(f"\n{pdf_file}:")
        for job in jobs:
            logger.info(f"  批次 {job['batch_index']}: {job['status']} (页 {job['page_start']}-{job['page_end']})")


def cleanup_all_batch_jobs(job_manager: BatchJobManager, client, logger: logging.Logger) -> Dict[str, int]:
    """取消并删除状态文件中记录的所有 Batch Job。"""
    all_jobs = job_manager.get_all_jobs()
    stats = {'total': len(all_jobs), 'cancelled': 0, 'deleted': 0, 'errors': 0}
    lock = None

    if not all_jobs:
        logger.info("没有可清理的 Batch Job 记录")
        return stats

    logger.info(f"开始清理 {len(all_jobs)} 个 Batch Job")

    for pdf_file, job in all_jobs:
        status = job.get('status', '')
        job_name = job.get('job_name')
        if not job_name:
            continue

        if status in BatchJobManager.ACTIVE_STATES:
            try:
                client.batches.cancel(name=job_name)
                stats['cancelled'] += 1
                logger.info(f"已取消: {job_name} ({pdf_file})")
            except Exception as e:
                stats['errors'] += 1
                logger.warning(f"取消失败 {job_name}: {e}")

    if stats['cancelled'] > 0:
        logger.info("等待 2 秒，让取消请求先到达服务端")
        time.sleep(2)

    from threading import Lock
    lock = Lock()

    def delete_one(pdf_file: str, job_name: str):
        try:
            client.batches.delete(name=job_name)
            with lock:
                stats['deleted'] += 1
            logger.info(f"已删除: {job_name} ({pdf_file})")
        except Exception as e:
            with lock:
                stats['errors'] += 1
            logger.warning(f"删除失败 {job_name}: {e}")

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(all_jobs)))) as executor:
        futures = []
        for pdf_file, job in all_jobs:
            job_name = job.get('job_name')
            if not job_name:
                continue
            futures.append(executor.submit(delete_one, pdf_file, job_name))
        for future in as_completed(futures):
            future.result()

    if stats['deleted'] == stats['total']:
        job_manager.jobs = {}
        job_manager._save()

    logger.info(
        f"清理完成: 总计 {stats['total']}，取消 {stats['cancelled']}，"
        f"删除 {stats['deleted']}，错误 {stats['errors']}"
    )
    return stats


BATCH_PRICING = {
    "gemini-3.1-pro-preview": {
        "tiers": [
            {"max_prompt_tokens": 200_000, "input": 1.00, "output": 6.00},
            {"input": 2.00, "output": 9.00},
        ]
    },
    "gemini-3.1-flash-lite-preview": {"input": 0.125, "output": 0.75},
    "gemini-3-flash-preview": {"input": 0.25, "output": 1.50},
    "gemini-2.5-pro": {
        "tiers": [
            {"max_prompt_tokens": 200_000, "input": 0.625, "output": 5.00},
            {"input": 1.25, "output": 7.50},
        ]
    },
    "gemini-2.5-flash": {"input": 0.15, "output": 1.25},
    "gemini-2.5-flash-lite": {"input": 0.05, "output": 0.20},
}


def _optional_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return int(raw)


def _resolve_batch_pricing(model_name: str, prompt_tokens: int) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    pricing = BATCH_PRICING.get(model_name)
    if not pricing:
        return None, None

    tiers = pricing.get("tiers")
    if not tiers:
        return pricing, None

    for tier in tiers:
        max_prompt_tokens = tier.get("max_prompt_tokens")
        if max_prompt_tokens is None or prompt_tokens <= max_prompt_tokens:
            tier_label = None
            if max_prompt_tokens is not None:
                tier_label = f"prompt <= {max_prompt_tokens:,}"
            else:
                tier_label = "prompt > 200,000"
            return {
                "input": float(tier["input"]),
                "output": float(tier["output"]),
            }, tier_label

    return None, None


def _log_usage_stats(logger: logging.Logger, usage: Dict[str, int], model_name: str):
    prompt_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('candidates_tokens', 0)
    thinking_tokens = usage.get('thinking_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    request_count = usage.get('request_count', 0)
    billed_output_tokens = output_tokens + thinking_tokens
    
    logger.info("=" * 60)


def save_batch_markdown(
    processor: GeminiOCRProcessor,
    pdf_path: Path,
    job_manager: BatchJobManager,
    config: Config,
    logger: logging.Logger,
):
    jobs = job_manager.get_all_jobs_for_pdf(pdf_path.name)
    if not jobs:
        logger.warning(f"{pdf_path.name}: 无 Batch Job 记录")
        return

    total_pages = sum(j['page_end'] - j['page_start'] + 1 for j in jobs)
    markdown_content, usage = processor.collect_batch_results(pdf_path, job_manager, total_pages)

    output_path = config.output_dir / f"{pdf_path.stem}.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write(f"*OCR processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"*Model: {config.model_name} (Batch API)*\n\n")
        f.write("---\n\n")
        f.write(markdown_content)

    logger.info(f"已保存: {output_path}")
    _log_usage_stats(logger, usage, config.model_name)


def run_batch_automation(
    processor: GeminiOCRProcessor,
    job_manager: BatchJobManager,
    pdf_files: List[Path],
    config: Config,
    logger: logging.Logger,
):
    remaining = list(pdf_files)

    while True:
        job_manager.sync_from_remote(processor.client)
        active_before = len(job_manager.get_active_jobs())
        limit_label = str(config.max_active_batch_jobs) if config.max_active_batch_jobs else "未启用"
        logger.info(
            f"自动调度轮次开始: 剩余 PDF {len(remaining)} 个，当前活跃 Batch Job {active_before}/{limit_label}。"
        )

        while remaining and (
            config.max_active_batch_jobs is None
            or len(job_manager.get_active_jobs()) < config.max_active_batch_jobs
        ):
            pdf_path = remaining[0]
            _, fully_submitted = processor.submit_pdf_batch_jobs(
                pdf_path,
                job_manager,
                allow_wait_for_slot=False,
            )
            if fully_submitted:
                remaining.pop(0)
            else:
                break

        job_manager.sync_from_remote(processor.client)
        active_count = len(job_manager.get_active_jobs())

        if not remaining and active_count == 0:
            logger.info("自动调度完成: 无剩余 PDF，且无活跃 Batch Job。")
            break

        logger.info(
            f"自动调度中: 剩余 PDF {len(remaining)} 个，活跃 Batch Job {active_count} 个，"
            f"等待 {config.poll_interval} 秒后继续。"
        )
        time.sleep(config.poll_interval)

    for pdf_path in pdf_files:
        save_batch_markdown(processor, pdf_path, job_manager, config, logger)
    logger.info("Token 使用统计")
    logger.info(f"  请求数: {request_count}")
    logger.info(f"  Input tokens:    {prompt_tokens:,}")
    logger.info(f"  Output tokens:   {output_tokens:,}")
    if thinking_tokens > 0:
        logger.info(f"  Thinking tokens: {thinking_tokens:,}")
    logger.info(f"  Total tokens:    {total_tokens:,}")
    
    pricing, tier_label = _resolve_batch_pricing(model_name, prompt_tokens)
    if pricing:
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (billed_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        logger.info(f"  估算成本 (Batch API 半价):")
        if tier_label:
            logger.info(f"    Pricing tier: {tier_label}")
        logger.info(f"    Input:    ${input_cost:.4f}")
        logger.info(f"    Output:   ${output_cost:.4f}")
        if thinking_tokens > 0:
            logger.info(f"    Billed output tokens (含 thinking): {billed_output_tokens:,}")
        logger.info(f"    Total:    ${total_cost:.4f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Gemini OCR - 支持实时和 Batch API 模式")
    parser.add_argument("--reset", action="store_true", help="重置处理状态，从头开始")
    parser.add_argument("--file", type=str, help="只处理指定的PDF文件")
    parser.add_argument("--realtime", action="store_true", help="使用实时 API（原有逻辑）")
    parser.add_argument("--no-wait", action="store_true", help="Batch 模式: 提交后立即退出")
    parser.add_argument("--batch-size", type=int, help="每批最大页数 (默认: 50)")
    parser.add_argument("--status", action="store_true", help="显示所有 Batch Job 状态")
    parser.add_argument("--download", action="store_true", help="下载已完成的 Batch 结果并合并")
    parser.add_argument("--cleanup-all-jobs", action="store_true", help="取消并删除状态文件中记录的所有 Batch Job")
    parser.add_argument("--max-active-batch-jobs", type=int, help="限制同时处于 pending/running 的 Batch Job 数")
    args = parser.parse_args()
    
    config = Config()
    
    if args.realtime:
        config.use_batch_api = False
    if args.no_wait:
        config.wait_for_completion = False
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_active_batch_jobs:
        config.max_active_batch_jobs = args.max_active_batch_jobs
    
    logger = setup_logging(config.logs_dir)
    
    if not config.validate():
        logger.error("配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    if args.status:
        job_manager = BatchJobManager(config.batch_status_file)
        client = genai.Client(api_key=config.gemini_api_key)
        logger.info("同步远程状态...")
        sync_result = job_manager.sync_from_remote(client)
        if sync_result['synced'] > 0:
            logger.info(f"已同步 {sync_result['synced']} 个 job 状态")
        print_job_status(job_manager, logger)
        sys.exit(0)

    if args.cleanup_all_jobs:
        job_manager = BatchJobManager(config.batch_status_file)
        client = genai.Client(api_key=config.gemini_api_key)
        cleanup_all_batch_jobs(job_manager, client, logger)
        sys.exit(0)
    
    mode = "实时 API" if not config.use_batch_api else "Batch API"
    logger.info("=" * 60)
    logger.info(f"Gemini OCR 启动 [{mode}]")
    logger.info(f"模型: {config.model_name}")
    logger.info(f"主语言: {config.primary_language}")
    if config.use_batch_api:
        logger.info(f"每批页数: {config.batch_size}")
        if config.max_active_batch_jobs:
            logger.info(f"最大活跃 Batch Job 数: {config.max_active_batch_jobs}")
        else:
            logger.info("最大活跃 Batch Job 数: 未启用")
        logger.info(f"等待完成: {'是' if config.wait_for_completion else '否'}")
    logger.info(f"输入目录: {config.input_dir}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info("=" * 60)
    
    if args.reset:
        if config.use_batch_api and config.batch_status_file.exists():
            config.batch_status_file.unlink()
            logger.info("已重置 Batch Job 状态")
        if config.status_file.exists():
            config.status_file.unlink()
            logger.info("已重置处理状态")
    
    if args.file:
        pdf_files = [config.input_dir / args.file]
        if not pdf_files[0].exists():
            logger.error(f"文件不存在: {args.file}")
            sys.exit(1)
    else:
        pdf_files = find_pdf_files(config.input_dir)
    
    if not pdf_files:
        logger.warning(f"未找到PDF文件，请将PDF放入 {config.input_dir}")
        sys.exit(0)
    
    logger.info(f"发现 {len(pdf_files)} 个PDF文件")
    
    processor = GeminiOCRProcessor(config)
    
    if config.use_batch_api:
        job_manager = BatchJobManager(config.batch_status_file)
        
        if args.download:
            logger.info("下载模式: 获取已完成的结果")
            logger.info("同步远程状态...")
            sync_result = job_manager.sync_from_remote(processor.client)
            if sync_result['synced'] > 0:
                logger.info(f"已同步 {sync_result['synced']} 个 job 状态")
            
            for pdf_path in pdf_files:
                save_batch_markdown(processor, pdf_path, job_manager, config, logger)
        else:
            if config.wait_for_completion:
                run_batch_automation(processor, job_manager, pdf_files, config, logger)
            else:
                for pdf_path in pdf_files:
                    try:
                        processor.process_pdf_batch(pdf_path, job_manager)
                        if processor.queue_limit_reached:
                            logger.info("已达到活跃 Batch Job 上限，本轮停止继续提交新任务")
                            break
                    except Exception as e:
                        logger.error(f"处理 {pdf_path.name} 时出错: {e}")
                        continue
        
        print_job_status(job_manager, logger)
        
    else:
        status_manager = StatusManager(config.status_file)
        
        for pdf_path in pdf_files:
            try:
                markdown_content = processor.process_pdf(pdf_path, status_manager)
                
                output_path = config.output_dir / f"{pdf_path.stem}.md"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {pdf_path.stem}\n\n")
                    f.write(f"*OCR processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(f"*Model: {config.model_name}*\n\n")
                    f.write("---\n\n")
                    f.write(markdown_content)
                
                logger.info(f"已保存: {output_path}")
                
            except Exception as e:
                logger.error(f"处理 {pdf_path.name} 时出错: {e}")
                continue
        
        stats = status_manager.get_stats()
        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"成功: {stats['completed']} 页")
        logger.info(f"失败: {stats['failed']} 页")
        logger.info(f"输出目录: {config.output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
