# Gemini Batch PDF OCR

[中文文档](README.zh-CN.md)

A batch PDF OCR tool powered by Google Gemini 2.5 Pro, designed for converting scanned PDF documents into high-quality Markdown format.

## Overview

This tool leverages Gemini's powerful vision capabilities to accurately recognize complex document structures and multiple languages (with special optimization for Arabic and other RTL scripts).

**Key Advantages:**
- **Batch API Support**: Uses Google Gemini Batch API for **50% cost savings** compared to real-time API, ideal for large-scale document processing.
- **High-Precision OCR**: Utilizes Gemini 2.5 Pro's native multimodal capabilities to handle tables, headings, lists, and complex layouts.
- **Robustness**: Built-in checkpoint recovery, retry mechanism, and concurrency control.

---

## Features

- **Dual Processing Modes**:
    - **Batch Mode (Recommended)**: Asynchronous processing, 50% cheaper, supports parallel submission of large files.
    - **Real-time Mode**: Synchronous processing, instant results, suitable for quick conversion of small documents.
- **Multi-language Support**: Optimized for Arabic by default, with automatic RTL format preservation.
- **Smart Formatting**: Outputs structured Markdown including headings, tables, lists, and quotes.
- **Checkpoint Recovery**: Automatically saves progress; resumes from last failure point after interruption.
- **Flexible Configuration**: Customizable DPI, concurrency, retry strategies, and OCR prompts.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mustakshif/gemini-batch-ocr.git
   cd gemini-batch-ocr
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (Poppler)**:
   This tool uses `pdf2image`, which requires Poppler.
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
   - **Windows**: Download Poppler binaries and add the `bin` directory to your system PATH.

---

## Configuration

1. Copy the configuration template:
   ```bash
   cp config.example.env .env
   ```

2. Edit `.env` file with your settings:
   - `GEMINI_API_KEY`: Your Google AI Studio API key.
   - `MODEL_NAME`: Model to use, defaults to `gemini-2.5-pro`.
   - `PRIMARY_LANGUAGE`: Primary recognition language (e.g., Arabic, Chinese, English).
   - `BATCH_SIZE`: Pages per batch in Batch mode (recommended: 50).

---

## Usage

Place PDF files to process in the `input/` directory.

### 1. Standard Run (Batch API Mode)
Uses Batch API by default, automatically submits jobs, polls status, and downloads results.
```bash
python batch_ocr.py
```

### 2. Real-time API Mode
For small files requiring immediate results.
```bash
python batch_ocr.py --realtime
```

### 3. Async Processing Pipeline
For large-scale tasks, execute in separate steps:
- **Submit jobs and exit**:
  ```bash
  python batch_ocr.py --no-wait
  ```
- **Check job status**:
  ```bash
  python batch_ocr.py --status
  ```
- **Download and merge results**:
  ```bash
  python batch_ocr.py --download
  ```

### 4. Other Commands
- **Process specific file**:
  ```bash
  python batch_ocr.py --file example.pdf
  ```
- **Reset status (start fresh)**:
  ```bash
  python batch_ocr.py --reset
  ```
- **Custom batch size**:
  ```bash
  python batch_ocr.py --batch-size 30
  ```

---

## Workflow (Batch API)

```
[ PDF File ] 
      |
      v
[ Convert to Images (DPI:200) ] 
      |
      v
[ Generate JSONL Request File ] (grouped by BATCH_SIZE)
      |
      v
[ Upload to Gemini File API ]
      |
      v
[ Create Batch Job ] <--- 50% Cost Savings!
      |
      v
[ Poll Job Status ]
      |
      v
[ Download JSONL Results ]
      |
      v
[ Parse and Merge to Markdown ]
```

---

## FAQ

**Q: Why use Batch API?**
A: Besides 50% price discount, Batch API has higher quota limits, enabling processing of thousands of pages simultaneously without hitting rate limits.

**Q: Where are processed files saved?**
A: Final Markdown files are saved in `output/`. Logs and intermediate states are in `logs/`.

**Q: What if some pages fail OCR?**
A: The tool retries automatically. If still failing, an HTML comment with error info is inserted in the Markdown for manual review.

---

## Project Structure

```
gemini-batch-ocr/
├── input/              # Place PDF files here
├── output/             # Markdown output
├── logs/               # Processing logs and status
├── batch_ocr.py        # Main script
├── config.example.env  # Configuration template
└── requirements.txt    # Python dependencies
```
