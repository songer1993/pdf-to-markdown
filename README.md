# PDF to Markdown Converter

## Overview

A simple Python script for converting PDF documents to Markdown files. This tool was developed for personal use and may not handle all PDF formats or edge cases.

### 🌟 Features

- **Multiple Extraction Methods**
  - Primary conversion using Markitdown
  - Fallback PyPDF2 extraction
  - Optional xAI-powered title extraction

- **Text Cleaning**
  - Removes problematic Unicode characters
  - Preserves document structure
  - Handles various PDF formats

- **Intelligent Filename Generation**
  - Uses paper title for output filename
  - Sanitizes filenames to prevent errors

### 🛠 Prerequisites

- Python 3.8+
- Required libraries:
  - PyPDF2
  - markitdown
  - python-dotenv
  - requests
  - openai (optional)

### 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/songer1993/pdf-to-markdown.git
cd pdf-to-markdown
```
2. Create a virtual environment
```bash
conda create -n pdf2md python=3.8
conda activate pdf2md
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
## 🔧 Configuration

1. Create a `.env` file in the project root
```plaintext
XAI_API_KEY=your_api_key_here
```

### 💻 Usage
```bash
python pdf_to_md_converter.py /path/to/input/pdfs /path/to/output/markdown
```

### 🔍 Optional Arguments

- `input_dir`: Directory containing source PDF files
- `output_dir`: Directory for generated Markdown files

### 🛡 Error Handling

- Logs conversion attempts and failures
- Generates `pdf_conversion.log` for tracking

### 📋 Logging

Conversion results are logged to:
- Console
- `pdf_conversion.log`


### ⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.

