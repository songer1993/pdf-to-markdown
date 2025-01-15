import os
import re
import logging
import warnings
import argparse
import unicodedata
from pathlib import Path

# Third-party imports
import pypdf
import requests
from dotenv import load_dotenv
from markitdown import MarkItDown
import pdfminer.high_level
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('pdf_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Text processing utilities
def clean_text(text):
    """Clean and sanitize text to ensure valid UTF-8 encoding while preserving text structure"""
    cleaned_text = ''.join(
        char if (
            char in ['\n', '\r', '\t', ' '] or 
            (not unicodedata.combining(char) and 
             unicodedata.category(char) != 'Cs')
        ) else '' 
        for char in text
    )
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned_text)

def clean_filename(filename):
    """Clean filename to remove invalid characters"""
    cleaned = re.sub(r'[<>:"/\\|?*]', '', filename)
    cleaned = cleaned.replace(' ', '_')
    return cleaned[:255]

def extract_title_with_xai(pdf_path, xai_api_key):
    """Extract paper title using xAI"""
    try:
        # Extract first page text for title extraction
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            first_page_text = reader.pages[0].extract_text()
        
        # Prepare xAI API request
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {xai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta",
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract the title of the academic paper from the given text. Return only the title, without any additional text."
                    },
                    {
                        "role": "user", 
                        "content": first_page_text[:1000]
                    }
                ],
                "max_tokens": 50
            }
        )
        
        response_data = response.json()
        extracted_title = response_data['choices'][0]['message']['content'].strip()
        return extracted_title if extracted_title else pdf_path.stem
    
    except Exception:
        return pdf_path.stem

def fallback_pdf_extraction(pdf_path):
    """Fallback text extraction method using pypdf"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = "\n".join(page.extract_text() for page in reader.pages)
            return clean_text(text)
    except Exception as e:
        logger.error(f"Fallback extraction failed for {pdf_path}: {e}")
        return ""

def clean_text_with_xai(text, xai_api_key):
    """
    Use xAI to clean and structure the extracted text.
    Replaces basic text cleaning with AI-powered cleaning.
    Args:
        text: Raw text content
        xai_api_key: xAI API key
    Returns:
        Cleaned and structured text
    """
    try:
        if not xai_api_key:
            logger.warning("xAI API key not found. Falling back to basic cleaning.")
            return clean_text(text)

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
        def call_xai_api(content):
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {xai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-beta",
                    "messages": [
                        {
                            "role": "system",
                            "content": """
                            Clean and structure the academic text with these requirements:
                            1. Fix OCR errors and encoding issues
                            2. Remove invalid Unicode characters while preserving valid ones
                            3. Maintain proper paragraph structure and indentation
                            4. Keep all mathematical symbols, equations, and special characters
                            5. Preserve citations and references
                            6. Remove redundant whitespace and control characters
                            7. Ensure consistent line endings
                            8. Handle multi-language text appropriately
                            Return only the cleaned text.
                            """
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1  # Low temperature for consistent cleaning
                }
            )
            
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()

        # Process text in chunks
        max_chunk = 8000
        if len(text) > max_chunk:
            chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Cleaning chunk {i+1}/{len(chunks)} with xAI...")
                cleaned_chunks.append(call_xai_api(chunk))
            return "\n\n".join(cleaned_chunks)
        else:
            return call_xai_api(text)

    except Exception as e:
        logger.error(f"xAI cleaning failed: {e}. Falling back to basic cleaning.")
        return clean_text(text)

def convert_pdfs_to_md(input_dir: str, output_dir: str, compress_md: bool = False):
    """Convert PDF files to Markdown, with xAI cleaning and optional compression"""
    xai_key = os.getenv('XAI_API_KEY')
    markitdown = MarkItDown()
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return
    
    total_files = len(pdf_files)
    converted_files = 0
    skipped_files = 0
    
    for pdf_path in pdf_files:
        if xai_key:
            try:
                paper_title = extract_title_with_xai(pdf_path, xai_key)
                output_file = Path(output_dir) / f"{clean_filename(paper_title)}.md"
            except Exception:
                output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        else:
            output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        
        if output_file.exists():
            logger.info(f"✓ {pdf_path.name} already converted (skipping)")
            skipped_files += 1
            continue
            
        try:
            text_content = ""
            conversion_methods = [
                lambda: markitdown.convert(str(pdf_path)).text_content if hasattr(markitdown.convert(str(pdf_path)), 'text_content') else None,
                lambda: pdfminer.high_level.extract_text(str(pdf_path)),
                lambda: fallback_pdf_extraction(pdf_path)
            ]
            
            for method in conversion_methods:
                try:
                    text_content = method()
                    if text_content and text_content.strip():
                        break
                except Exception as method_error:
                    logger.warning(f"Conversion method failed for {pdf_path}: {method_error}")
            
            if not text_content:
                logger.warning(f"Skipping {pdf_path.name}: No text extracted")
                skipped_files += 1
                continue
            
            cleaned_text = clean_text_with_xai(text_content, xai_key)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            if compress_md:
                original_size = len(text_content.encode('utf-8'))
                compressed_size = len(cleaned_text.encode('utf-8'))
                reduction = (original_size - compressed_size) / original_size * 100
                logger.info(f"Markdown compression: {reduction:.1f}% reduction")
            
            logger.info(f"✓ {pdf_path.name} → {output_file.name}")
            converted_files += 1
        except Exception as e:
            logger.error(f"✗ {pdf_path.name}: {e}")
    
    logger.info(f"\nSummary:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Converted: {converted_files}")
    logger.info(f"Skipped (already converted): {skipped_files}")

# Main entry point
def main():
    parser = argparse.ArgumentParser(description='PDF to Markdown Converter')
    parser.add_argument('input_dir', help='Input PDF directory')
    parser.add_argument('output_dir', help='Output Markdown directory')
    args = parser.parse_args()
    convert_pdfs_to_md(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()