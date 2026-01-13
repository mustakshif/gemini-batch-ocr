# Gemini Batch PDF OCR 工具

[English](README.md)

这是一个基于 Google Gemini 3 Flash Preview 模型的批量 PDF OCR 工具，专门设计用于处理扫描版 PDF 文档并将其转换为高质量的 Markdown 格式。查看[基准测试结果](https://github.com/mustakshif/arabic-ocr-benchmark-tool?tab=readme-ov-file#benchmark-results)了解模型对比。

## 项目简介

本项目利用 Gemini 强大的视觉理解能力，能够精准识别复杂的文档结构和多种语言（特别优化了阿拉伯语等从右向左 RTL 的脚本）。

**核心优势：**
- **Batch API 支持**：支持 Google Gemini Batch API，相比实时 API 可**节省 50% 的成本**，非常适合大规模文档处理。
- **高精度 OCR**：利用 Gemini 3 Flash Preview 的原生多模态能力，能够处理表格、标题、列表等复杂布局。
- **稳健性**：内置断点续传、失败重试和并发控制机制。

---

### 功能特性

- **双模式运行**：
    - **Batch 模式（推荐）**：异步处理，成本减半，支持大规模文件并行提交。
    - **实时模式**：同步处理，即时获取结果，适合少量页面的快速转换。
- **多语言支持**：默认针对阿拉伯语优化，支持自动识别并保留 RTL 格式。
- **智能排版**：输出结构化的 Markdown，包括标题、表格、列表和引用。
- **断点续传**：自动记录处理进度，程序中断后重启可从上次失败的位置继续。
- **灵活配置**：可自定义 DPI、并发数、重试策略及 OCR Prompt。

---

### 安装步骤

1. **克隆项目**：
   ```bash
   git clone <repository-url>
   cd OCR-tool/gemini-batch-by-Claude
   ```

2. **安装 Python 依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **安装系统依赖 (Poppler)**：
   本工具使用 `pdf2image` 库，需要系统安装 Poppler。
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
   - **Windows**: 下载 Poppler 二进制文件并将其 `bin` 目录添加到系统 PATH。

---

### 配置说明

1. 复制配置模板：
   ```bash
   cp config.example.env .env
   ```

2. 编辑 `.env` 文件，填入必要信息：
   - `GEMINI_API_KEY`: 你的 Google AI Studio API 密钥。
   - `MODEL_NAME`: 使用的模型，默认为 `gemini-3-flash-preview`。
   - `PRIMARY_LANGUAGE`: 主要识别语言（如 Arabic, Chinese, English）。
   - `BATCH_SIZE`: Batch 模式下每批处理的页数（建议 50）。

---

### 使用方法

将需要处理的 PDF 文件放入 `input/` 目录。

#### 1. 标准运行 (Batch API 模式)
默认使用 Batch API，自动提交任务、轮询状态并下载结果。
```bash
python batch_ocr.py
```

#### 2. 实时 API 模式
适用于需要立即看到结果的小文件。
```bash
python batch_ocr.py --realtime
```

#### 3. 异步处理流水线
对于超大规模任务，可以分步执行：
- **提交任务并退出**：
  ```bash
  python batch_ocr.py --no-wait
  ```
- **查看任务状态**：
  ```bash
  python batch_ocr.py --status
  ```
- **手动下载并合并结果**：
  ```bash
  python batch_ocr.py --download
  ```

#### 4. 其他常用命令
- **处理特定文件**：
  ```bash
  python batch_ocr.py --file example.pdf
  ```
- **重置状态（从头开始）**：
  ```bash
  python batch_ocr.py --reset
  ```

---

### 工作流程说明 (Batch API)

```text
[ PDF 文件 ] 
      |
      v
[ 转换为图片 (DPI:200) ] 
      |
      v
[ 生成 JSONL 请求文件 ] (按 BATCH_SIZE 分组)
      |
      v
[ 上传至 Gemini File API ]
      |
      v
[ 创建 Batch Job ] <--- 成本降低 50% !
      |
      v
[ 轮询 Job 状态 (Poll) ]
      |
      v
[ 下载 JSONL 结果 ]
      |
      v
[ 解析并合并为 Markdown ]
```

---

### 常见问题

**Q: 为什么推荐使用 Batch API？**
A: 除了 50% 的价格优惠外，Batch API 拥有更高的配额限制，能够同时处理数千页文档而不会触发现流限制。

**Q: 处理后的文件保存在哪里？**
A: 最终的 Markdown 文件保存在 `output/` 目录。日志和中间状态保存在 `logs/` 目录。

**Q: 如果某几页 OCR 失败了怎么办？**
A: 工具会自动重试。如果最终依然失败，会在输出的 Markdown 中插入包含错误信息的 HTML 注释，方便后续人工核查。
