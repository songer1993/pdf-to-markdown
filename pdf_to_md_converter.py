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
import openai  # Changed to use OpenAI's SDK for DeepSeek compatibility
from dotenv import load_dotenv
from markitdown import MarkItDown
import pdfminer.high_level
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.utils import CryptographyDeprecationWarning

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

# Suppress HTTP request logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure OpenAI client for DeepSeek
client = openai.OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1"  # Using v1 endpoint for stability
)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

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

def extract_title_from_text(cleaned_text, fallback_name):
    """Extract paper title from cleaned text using DeepSeek"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise academic paper title extractor. Extract only the main title from the text, without any additional content. The text is already cleaned and formatted."
                },
                {
                    "role": "user",
                    "content": cleaned_text[:1000]  # Use first 1000 chars of cleaned text
                }
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        extracted_title = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        return extracted_title if extracted_title else fallback_name
    
    except Exception as e:
        logger.error(f"Title extraction failed: {e}")
        return fallback_name

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

def clean_text_with_deepseek(text):
    """Use DeepSeek to clean and structure the extracted text with context caching"""
    try:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
        def call_deepseek_api(content, is_continuation=False):
            system_prompt = """You are a text cleaner that preserves markdown formatting. Your task is to clean and structure academic text while maintaining all markdown syntax.
                Output ONLY the cleaned text without any explanations.
                Requirements:
                1. Preserve all markdown syntax including:
                   - Headers (#, ##, ###)
                   - Lists (* or -, numbered lists)
                   - Code blocks (```)
                   - Tables
                   - Bold and italic markers (* or _)
                   - Links and images
                2. Fix OCR errors and encoding issues
                3. Maintain proper paragraph structure
                4. Keep all mathematical symbols and equations
                5. Preserve citations and references
                6. Remove redundant whitespace
                7. Ensure consistent line endings"""
            
            if is_continuation:
                system_prompt = "Continue cleaning the text while preserving all markdown formatting. Output ONLY the cleaned text without any explanations."

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Update the usage logging to use standard OpenAI response attributes
            usage = response.usage
            if usage:
                logger.info(f"Total tokens used: {usage.total_tokens}")
                logger.info(f"Prompt tokens: {usage.prompt_tokens}")
                logger.info(f"Completion tokens: {usage.completion_tokens}")
            
            return response.choices[0].message.content.strip() if response.choices[0].message.content else ""

        # Process text in chunks optimized for DeepSeek's context window
        max_chunk = 12000  # Optimized for DeepSeek's 64K context window
        if len(text) > max_chunk:
            chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
            cleaned_chunks = []
            
            # Process chunks with context caching awareness
            for i, chunk in enumerate(chunks):
                logger.info(f"Cleaning chunk {i+1}/{len(chunks)} with DeepSeek...")
                if i == 0:
                    cleaned_chunks.append(call_deepseek_api(chunk))
                else:
                    # Use prefix from previous chunk to maintain context and leverage caching
                    overlap = 200
                    context = cleaned_chunks[-1][-overlap:] if cleaned_chunks else ""
                    cleaned_chunks.append(call_deepseek_api(context + chunk, is_continuation=True))
            
            return "\n\n".join(cleaned_chunks)
        else:
            return call_deepseek_api(text)

    except Exception as e:
        logger.error(f"DeepSeek cleaning failed: {e}. Falling back to basic cleaning.")
        return clean_text(text)

def convert_pdfs_to_md(input_dir: str, output_dir: str, compress_md: bool = False):
    """Convert PDF files to Markdown, with DeepSeek cleaning and optional compression"""
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
            
            cleaned_text = clean_text_with_deepseek(text_content)
            
            # Extract title from cleaned text
            paper_title = extract_title_from_text(cleaned_text, pdf_path.stem)
            output_file = Path(output_dir) / f"{clean_filename(paper_title)}.md"
            
            if output_file.exists():
                logger.info(f"✓ {pdf_path.name} already converted (skipping)")
                skipped_files += 1
                continue
            
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