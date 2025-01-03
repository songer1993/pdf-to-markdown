import os
import re
from dotenv import load_dotenv
import logging
import argparse
from pathlib import Path
from markitdown import MarkItDown
from openai import OpenAI
import pypdf
import unicodedata
import requests
import pdfminer.high_level

# Suppress pydub ffmpeg warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('pdf_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and sanitize text to ensure valid UTF-8 encoding while preserving text structure"""
    # Remove or replace problematic Unicode characters
    cleaned_text = ''.join(
        char if (
            # Keep line breaks, spaces, and other whitespace
            char in ['\n', '\r', '\t', ' '] or 
            # Remove combining characters and surrogate characters
            (not unicodedata.combining(char) and 
             unicodedata.category(char) != 'Cs')
        ) else '' 
        for char in text
    )
    
    # Remove non-printable control characters, except for line breaks and tabs
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned_text)
    
    return cleaned_text

def clean_filename(filename):
    """Clean filename to remove invalid characters"""
    # Remove or replace characters not allowed in filenames
    cleaned = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Limit filename length
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
                        "content": first_page_text[:1000]  # Limit text to first 1000 characters
                    }
                ],
                "max_tokens": 50
            }
        )
        
        # Parse response
        response_data = response.json()
        extracted_title = response_data['choices'][0]['message']['content'].strip()
        
        # Fallback to filename if title extraction fails
        return extracted_title if extracted_title else pdf_path.stem
    
    except Exception:
        return pdf_path.stem

def fallback_pdf_extraction(pdf_path):
    """Fallback text extraction method using pypdf"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            # Extract text and clean it
            text = "\n".join(page.extract_text() for page in reader.pages)
            return clean_text(text)
    except Exception as e:
        logger.error(f"Fallback extraction failed for {pdf_path}: {e}")
        return ""
    
# def convert_pdfs_to_md(input_dir: str, output_dir: str):
    """Convert PDF files to Markdown"""
    # Retrieve xAI API key
    xai_key = os.getenv('XAI_API_KEY')
    
    # Initialize Markitdown
    markitdown = MarkItDown()
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return
    
    total_files = len(pdf_files)
    converted_files = 0
    failed_files = 0
    
    for pdf_path in pdf_files:
        # Determine output filename
        if xai_key:
            try:
                paper_title = extract_title_with_xai(pdf_path, xai_key)
                output_file = Path(output_dir) / f"{clean_filename(paper_title)}.md"
            except Exception:
                output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        else:
            output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        
        try:
            # Primary conversion attempt
            text_content = ""
            try:
                # Attempt Markitdown conversion
                result = markitdown.convert(str(pdf_path))
                
                # Explicitly check for text_content
                if hasattr(result, 'text_content'):
                    text_content = result.text_content
                else:
                    raise ValueError("No text content found in Markitdown result")
            
            except Exception as convert_error:
                logger.warning(f"Markitdown conversion failed: {convert_error}")
                
                # Fallback to PyPDF2 extraction
                text_content = fallback_pdf_extraction(pdf_path)
            
            # Ensure we have some content
            if not text_content:
                raise ValueError("No text extracted from PDF")
            
            # Clean and write markdown file
            cleaned_text = clean_text(text_content)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            logger.info(f"✓ {pdf_path.name} → {output_file.name}")
            converted_files += 1
        
        except Exception as final_error:
            logger.error(f"✗ {pdf_path.name}: {final_error}")
            failed_files += 1
    
    # Minimal summary
    logger.info(f"\nSummary: {converted_files}/{total_files} converted")

def is_valid_pdf(pdf_path):
    """
    Check if the PDF file is valid and not corrupted
    
    Args:
        pdf_path (Path): Path to the PDF file
    
    Returns:
        bool: True if PDF is valid, False otherwise
    """
    try:
        # Try to open the PDF and read its pages
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            # Check if the PDF has at least one page
            if len(reader.pages) == 0:
                logger.warning(f"Empty PDF: {pdf_path.name}")
                return False
            
            # Try to extract text from the first page as a basic integrity check
            reader.pages[0].extract_text()
            
            return True
    except Exception as e:
        logger.warning(f"Unexpected error checking PDF {pdf_path.name}: {e}")
        return False

def convert_pdfs_to_md(input_dir: str, output_dir: str):
    """Convert PDF files to Markdown, skipping already converted files"""
    # Retrieve xAI API key
    xai_key = os.getenv('XAI_API_KEY')
    
    # Initialize Markitdown
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
        # Determine output filename
        if xai_key:
            try:
                paper_title = extract_title_with_xai(pdf_path, xai_key)
                output_file = Path(output_dir) / f"{clean_filename(paper_title)}.md"
            except Exception:
                output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        else:
            output_file = Path(output_dir) / f"{pdf_path.stem}.md"
        
        # Skip if file already exists
        if output_file.exists():
            logger.info(f"✓ {pdf_path.name} already converted (skipping)")
            skipped_files += 1
            continue
        
        # Rest of the existing conversion logic remains the same
        try:
            # Primary conversion attempt
            text_content = ""
            conversion_methods = [
                # Method 1: Markitdown
                lambda: markitdown.convert(str(pdf_path)).text_content if hasattr(markitdown.convert(str(pdf_path)), 'text_content') else None,
                
                # Method 2: pdfminer extraction
                lambda: pdfminer.high_level.extract_text(str(pdf_path)),
                
                # Method 3: Fallback pypdf extraction
                lambda: fallback_pdf_extraction(pdf_path)
            ]
            
            # Try conversion methods sequentially
            for method in conversion_methods:
                try:
                    text_content = method()
                    if text_content and text_content.strip():
                        break
                except Exception as method_error:
                    logger.warning(f"Conversion method failed for {pdf_path}: {method_error}")
            
            # Ensure we have some content
            if not text_content:
                logger.warning(f"Skipping {pdf_path.name}: No text extracted")
                skipped_files += 1
                continue
            
            # Clean and write markdown file
            cleaned_text = clean_text(text_content)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            logger.info(f"✓ {pdf_path.name} → {output_file.name}")
            converted_files += 1
        except Exception as e:
            logger.error(f"✗ {pdf_path.name}: {e}")
    
    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Converted: {converted_files}")
    logger.info(f"Skipped (already converted): {skipped_files}")

def main():
    parser = argparse.ArgumentParser(description='PDF to Markdown Converter')
    parser.add_argument('input_dir', help='Input PDF directory')
    parser.add_argument('output_dir', help='Output Markdown directory')
    
    args = parser.parse_args()
    convert_pdfs_to_md(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()