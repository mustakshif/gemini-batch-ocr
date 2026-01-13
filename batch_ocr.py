#!/usr/bin/env python3
"""
Gemini 2.5 Pro Batch OCR Script
================================
批量处理扫描版PDF，使用Gemini 2.5 Pro进行OCR，输出Markdown格式。

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
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-pro")
        
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
        self.wait_for_completion = True  # 是否等待完成
        
        # Batch API 目录
        self.batch_requests_dir = self.logs_dir / "batch_requests"
        self.batch_results_dir = self.logs_dir / "batch_results"
        self.batch_status_file = self.logs_dir / "batch_jobs.json"
        
        # 语言配置
        self.primary_language = os.getenv("PRIMARY_LANGUAGE", "Arabic")
        self.ocr_prompt = os.getenv("OCR_PROMPT", self._default_prompt())
        
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
            if job['status'] in ('pending', 'JOB_STATE_PENDING', 'JOB_STATE_RUNNING'):
                pending.append(job)
        return pending
    
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


# ============================================================================
# OCR处理器
# ============================================================================

class GeminiOCRProcessor:
    """Gemini OCR处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.client = genai.Client(api_key=config.gemini_api_key)
        
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
                
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=self.config.ocr_prompt),
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
        return {
            "key": f"{safe_pdf_id}_page_{page_num:04d}",
            "request": {
                "contents": [{
                    "parts": [
                        {"text": self.config.ocr_prompt},
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
    
    def parse_batch_results(self, jsonl_content: str) -> Dict[str, str]:
        results = {}
        for line in jsonl_content.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                key = data.get('key', '')
                if 'response' in data and data['response']:
                    candidates = data['response'].get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts:
                            text = parts[0].get('text', '')
                            results[key] = text
                elif 'error' in data:
                    results[key] = f"<!-- OCR Error: {data['error']} -->"
            except json.JSONDecodeError as e:
                self.logger.warning(f"解析结果行失败: {e}")
        return results
    
    def process_pdf_batch(self, pdf_path: Path, job_manager: BatchJobManager) -> Optional[str]:
        pdf_name = pdf_path.name
        safe_pdf_id = self._safe_ascii_name(pdf_path.stem)
        self.logger.info(f"[Batch模式] 开始处理: {pdf_name}")
        
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)
        
        batch_size = self.config.batch_size
        num_batches = (total_pages + batch_size - 1) // batch_size
        
        self.logger.info(f"共 {total_pages} 页，分 {num_batches} 批处理 (每批 {batch_size} 页)")
        
        submitted_jobs = []
        
        for batch_idx in range(num_batches):
            if job_manager.is_batch_completed(pdf_name, batch_idx):
                self.logger.info(f"批次 {batch_idx + 1}/{num_batches} 已完成，跳过")
                continue
            
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
            submitted_jobs.append(job_name)
        
        if not self.config.wait_for_completion:
            self.logger.info("提交完成，--no-wait 模式，退出等待")
            self.logger.info(f"稍后使用 --download 获取结果")
            return None
        
        if not submitted_jobs:
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
                              total_pages: int) -> str:
        pdf_name = pdf_path.name
        safe_pdf_id = self._safe_ascii_name(pdf_path.stem)
        all_results: Dict[int, str] = {}
        
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
            
            parsed = self.parse_batch_results(jsonl_content)
            
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
        
        return "\n\n---\n\n".join(full_content)


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


def main():
    parser = argparse.ArgumentParser(description="Gemini OCR - 支持实时和 Batch API 模式")
    parser.add_argument("--reset", action="store_true", help="重置处理状态，从头开始")
    parser.add_argument("--file", type=str, help="只处理指定的PDF文件")
    parser.add_argument("--realtime", action="store_true", help="使用实时 API（原有逻辑）")
    parser.add_argument("--no-wait", action="store_true", help="Batch 模式: 提交后立即退出")
    parser.add_argument("--batch-size", type=int, help="每批最大页数 (默认: 50)")
    parser.add_argument("--status", action="store_true", help="显示所有 Batch Job 状态")
    parser.add_argument("--download", action="store_true", help="下载已完成的 Batch 结果并合并")
    args = parser.parse_args()
    
    config = Config()
    
    if args.realtime:
        config.use_batch_api = False
    if args.no_wait:
        config.wait_for_completion = False
    if args.batch_size:
        config.batch_size = args.batch_size
    
    logger = setup_logging(config.logs_dir)
    
    if not config.validate():
        logger.error("配置验证失败，请检查 .env 文件")
        sys.exit(1)
    
    if args.status:
        job_manager = BatchJobManager(config.batch_status_file)
        print_job_status(job_manager, logger)
        sys.exit(0)
    
    mode = "实时 API" if not config.use_batch_api else "Batch API"
    logger.info("=" * 60)
    logger.info(f"Gemini OCR 启动 [{mode}]")
    logger.info(f"模型: {config.model_name}")
    logger.info(f"主语言: {config.primary_language}")
    if config.use_batch_api:
        logger.info(f"每批页数: {config.batch_size}")
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
            for pdf_path in pdf_files:
                jobs = job_manager.get_all_jobs_for_pdf(pdf_path.name)
                if not jobs:
                    logger.warning(f"{pdf_path.name}: 无 Batch Job 记录")
                    continue
                
                total_pages = sum(j['page_end'] - j['page_start'] + 1 for j in jobs)
                markdown_content = processor.collect_batch_results(pdf_path, job_manager, total_pages)
                
                output_path = config.output_dir / f"{pdf_path.stem}.md"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {pdf_path.stem}\n\n")
                    f.write(f"*OCR processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                    f.write(f"*Model: {config.model_name} (Batch API)*\n\n")
                    f.write("---\n\n")
                    f.write(markdown_content)
                
                logger.info(f"已保存: {output_path}")
        else:
            for pdf_path in pdf_files:
                try:
                    result = processor.process_pdf_batch(pdf_path, job_manager)
                    
                    if result:
                        output_path = config.output_dir / f"{pdf_path.stem}.md"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {pdf_path.stem}\n\n")
                            f.write(f"*OCR processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                            f.write(f"*Model: {config.model_name} (Batch API)*\n\n")
                            f.write("---\n\n")
                            f.write(result)
                        logger.info(f"已保存: {output_path}")
                    
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
