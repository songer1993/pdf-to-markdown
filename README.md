# PDF to Markdown Converter

## Overview

A Python script for converting academic PDF papers to structured Markdown files, with intelligent text cleaning and formatting using DeepSeek AI.

### 🌟 Features

- **Multiple Extraction Methods**
  - Primary: MarkItDown for robust text extraction
  - Secondary: PDFMiner for complex PDFs
  - Fallback: PyPDF for basic extraction

- **AI-Powered Processing**
  - DeepSeek-based text cleaning and structuring
  - Intelligent title extraction
  - Handles large documents in chunks

- **Smart Text Processing**
  - Preserves academic paper structure
  - Maintains markdown formatting
  - Cleans OCR errors and artifacts
  - Handles Unicode and special characters

- **Parallel Processing**
  - Multi-threaded PDF conversion
  - Graceful interrupt handling
  - Progress tracking and logging

### 🛠 Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt
- DeepSeek API key for AI features

### 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/songer1993/pdf-to-markdown.git
   cd pdf-to-md-converter
   ```

2. Create and activate conda environment:
   ```bash
   conda create -n pdf2md python=3.8
   conda activate pdf2md
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create .env file:
   ```bash
   touch .env
   ```

5. Add your DeepSeek API key to .env:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

### 📖 Usage

1. Basic usage with command line:
   ```bash
   python pdf_to_md_converter.py input_directory output_directory
   ```

2. Example:
   ```bash
   python pdf_to_md_converter.py papers/ markdown/
   ```

The script will:
- Process all PDF files in the input directory
- Create markdown files in the output directory
- Extract and clean text using DeepSeek AI
- Generate filenames based on paper titles
- Skip existing files to avoid duplicates
