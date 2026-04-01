# Gemini/Vertex Batch PDF OCR 工具

[English](README.md)

这是一个批量 PDF OCR 工具，当前提供两套后端脚本：

- Gemini Developer API：`batch_ocr.py`
- Vertex AI：`vertex_ocr.py`

两套脚本都可以把扫描 PDF 转为 Markdown，并支持实时与批量流程。查看[基准测试结果](https://github.com/mustakshif/arabic-ocr-benchmark-tool?tab=readme-ov-file#benchmark-results)了解模型对比。

## 项目简介

本项目利用 Gemini 强大的视觉理解能力，能够精准识别复杂的文档结构和多种语言（特别优化了阿拉伯语等从右向左 RTL 的脚本）。

**核心优势：**
- **双后端并行**：保留原有 Gemini Developer 路线，同时提供 Vertex 路线，迁移风险更低。
- **Batch 成本优势**：两套后端都支持 Batch 工作流，适合非实时的大规模 OCR。
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
   - `MODEL_NAME`: 使用的模型，默认为 `gemini-3.1-flash-lite-preview`。
   - `PRIMARY_LANGUAGE`: 主要识别语言（如 Arabic, Chinese, English）。
   - `BATCH_SIZE`: Batch 模式下每批处理的页数（建议 50）。
   - `MAX_ACTIVE_BATCH_JOBS`: 可选参数，用来限制同时处于 `PENDING/RUNNING` 的 Batch Job 数量上限。默认不启用；需要时可设为 10-30。

3. 如果要使用 `vertex_ocr.py`，还需要配置：
   - `GOOGLE_CLOUD_PROJECT`
   - `GOOGLE_CLOUD_LOCATION`（例如 `global`）
   - `VERTEX_MODEL_NAME`（例如 `gemini-2.5-flash-lite`）
   - `VERTEX_GCS_BUCKET`
   - `VERTEX_GCS_PREFIX`（可选）
   - `VERTEX_API_KEY`（可选，仅实时模式）

Vertex 鉴权注意事项：
- 实时模式可以用 `VERTEX_API_KEY`。
- Batch 模式会忽略 `VERTEX_API_KEY`，改走 `ADC + project/location`。
- Batch 模式还需要项目 IAM 权限（如 `aiplatform.batchPredictionJobs.create`）和 GCS 读写权限。

### 默认模型与 Batch 定价

下面价格已在 **2026-03-21** 按 Gemini Developer API 官方页面核对：
<https://ai.google.dev/gemini-api/docs/pricing>

| 模型 | Batch 输入 / 100万 tokens | Batch 输出 / 100万 tokens | 说明 |
| --- | ---: | ---: | --- |
| `gemini-3.1-flash-lite-preview` | $0.125 | $0.75 | 默认模型，当前成本最低 |
| `gemini-3-flash-preview` | $0.25 | $1.50 | 更强一些，但更贵 |
| `gemini-2.5-flash` | $0.15 | $1.25 | 稳定版，速度和成本较平衡 |
| `gemini-2.5-flash-lite` | $0.05 | $0.20 | 最便宜的稳定版 |
| `gemini-2.5-pro` | $0.625 / $1.25 | $5.00 / $7.50 | 输入超过 200k tokens 时使用高档位 |
| `gemini-3.1-pro-preview` | $1.00 / $2.00 | $6.00 / $9.00 | 输入超过 200k tokens 时使用高档位 |

说明：
- Google 当前把 thinking tokens 计入输出价格，这个项目的成本估算也已按这个口径更新。
- Batch API 的价格通常约为标准 API 的 50%。

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
  现在这个模式会采用“补位提交”策略：只补到 `MAX_ACTIVE_BATCH_JOBS` 上限就退出，不再一次性把所有文件全提交上去。后面再运行一次即可继续补交剩余任务。
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
- **限制本轮活跃 Batch Job 数**：
  ```bash
  python batch_ocr.py --no-wait --max-active-batch-jobs 15
  ```
- **取消并删除当前记录里的所有 Batch Job**：
  ```bash
  python batch_ocr.py --cleanup-all-jobs
  ```

---

## Vertex 用法（`vertex_ocr.py`）

同样把待处理 PDF 放到 `input/` 目录。

### 1. 标准运行（Vertex Batch）
```bash
python vertex_ocr.py
```

### 2. 实时模式
```bash
python vertex_ocr.py --realtime
```

### 3. 异步流程
```bash
python vertex_ocr.py --no-wait
python vertex_ocr.py --status
python vertex_ocr.py --download
```

### 4. 清理命令
- **取消并删除本地记录中的 Vertex Batch Job 资源**：
  ```bash
  python vertex_ocr.py --cleanup-all-jobs
  ```
- **删除本地记录对应的 GCS 输入/输出产物**：
  ```bash
  python vertex_ocr.py --cleanup-gcs-artifacts
  ```
- **只清理单个 PDF 的 GCS 产物**：
  ```bash
  python vertex_ocr.py --file example.pdf --cleanup-gcs-artifacts
  ```

说明：
- `--cleanup-all-jobs` 只删除 Vertex Batch Job 资源。
- `--cleanup-gcs-artifacts` 删除 GCS 对象（按本地状态记录里的前缀）。
- 任务完成后 GCS 文件不会自动清理；需要手动清理或配置 bucket 生命周期规则。

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

**Q: 为什么 `--no-wait` 现在不再一次性提交完所有任务？**
A: Gemini Batch API 本身有并发 Batch Job 上限，系统负载高时队列等待也会明显变长。把活跃任务数控住，会比“一把全交”更稳。

**Q: 这个脚本现在怎么估算成本？**
A: 脚本会按上面的 Batch 价格表估算，并把 thinking tokens 一起计入 output 成本，和当前 Gemini 官方定价口径保持一致。

**Q: 处理后的文件保存在哪里？**
A: 最终的 Markdown 文件保存在 `output/` 目录。日志和中间状态保存在 `logs/` 目录。

**Q: 如果某几页 OCR 失败了怎么办？**
A: 工具会自动重试。如果最终依然失败，会在输出的 Markdown 中插入包含错误信息的 HTML 注释，方便后续人工核查。

**Q: Vertex 版本是不是也在用 Batch API？**
A: 是。`vertex_ocr.py` 的 Batch 路径走的是 Vertex Batch Job（`client.batches.create`），输入输出基于 GCS。

**Q: Vertex Batch 的 GCS 输出会自动删吗？**
A: 不会。请用 `--cleanup-gcs-artifacts`，或者在 Cloud Storage 配置生命周期规则自动删除。
