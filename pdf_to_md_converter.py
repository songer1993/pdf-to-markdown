import os
import re
import logging
import warnings
import argparse
import unicodedata
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import pypdf
import requests
import openai  # Changed to use OpenAI's SDK for DeepSeek compatibility
from dotenv import load_dotenv
from markitdown import MarkItDown
import pdfminer.high_level
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.utils import CryptographyDeprecationWarning
from tqdm import tqdm

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

# Constants section - group related constants
API_CONFIG = {
    'model': 'deepseek-chat',
    'max_tokens': 4000,
    'temperature': 0.1,
    'base_url': 'https://api.deepseek.com/v1'
}

FILE_PROCESSING = {
    'max_chunk_size': 12000,
    'max_filename_length': 255,
    'max_workers': 32,
    'overlap_size': 200
}

# System prompts as constants
TITLE_EXTRACTION_PROMPT = """You are a precise academic paper title extractor. 
Extract only the main title from the text, without any additional content. 
The text is already cleaned and formatted."""

TEXT_CLEANING_PROMPT = """You are a text cleaner specialized in academic papers. Clean and structure the text while:

1. KEEP:
   - Main title
   - Author names
   - Abstract
   - Section headings and content
   - Key findings and conclusions
   - Important figures and tables
   - Essential equations and formulas
   - Relevant citations in-text

2. REMOVE:
   - Author affiliations and contact details
   - Headers and footers
   - Page numbers
   - Copyright notices and journal information
   - Acknowledgments section
   - Funding statements
   - Conference presentation details
   - Declarations of interest
   - Running headers/footers

3. FORMAT:
   - Use markdown headers (#, ##, ###) for section titles
   - Use lists (* or -) for bullet points
   - Use code blocks (```) ONLY for actual code snippets or algorithms
   - Keep tables in markdown format
   - Use inline code (`) for short code references
   - Preserve mathematical notation
   - Maintain logical paragraph breaks
   - Clean up OCR errors
   - Remove duplicate whitespace

Output ONLY the cleaned text without explanations."""

CONTINUATION_PROMPT = """Continue cleaning the text while preserving all markdown formatting. 
Output ONLY the cleaned text without any explanations."""

# Global flag for graceful interruption
interrupted = False

# Move API client configuration to a separate function
def configure_api_client():
    """Configure and return OpenAI client for DeepSeek"""
    return openai.OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com/v1"
    )
# After the constants section, initialize the OpenAI client
client = configure_api_client()

# Suppress warnings (add back after constants)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global interrupted
    if not interrupted:
        logger.info("\nGracefully shutting down... (Press Ctrl+C again to force quit)")
        interrupted = True
    else:
        logger.info("\nForce quitting...")
        exit(1)

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

# Improve text cleaning function
def clean_text(text):
    """Clean and sanitize text to ensure valid UTF-8 encoding while preserving text structure"""
    if not text:
        return ""
        
    # First pass: Remove control characters except newlines and whitespace
    cleaned = ''.join(
        char for char in text
        if char in ['\n', '\r', '\t', ' '] or 
        (not unicodedata.combining(char) and unicodedata.category(char) != 'Cs')
    )
    
    # Second pass: Remove specific control characters
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
    
    # Remove excessive whitespace while preserving structure
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    
    return cleaned.strip()

# Improve filename cleaning
def clean_filename(filename):
    """Clean filename to remove invalid characters and ensure proper length"""
    if not filename:
        return "untitled"
        
    # Remove invalid characters and replace spaces
    cleaned = re.sub(r'[<>:"/\\|?*]', '', filename)
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Ensure proper length and remove trailing periods
    cleaned = cleaned[:FILE_PROCESSING['max_filename_length']].rstrip('.')
    
    return cleaned or "untitled"

# Update extraction methods to be more maintainable
def extract_text_using_markitdown(pdf_path):
    """Extract text using MarkItDown"""
    return MarkItDown().convert(str(pdf_path)).text_content

def extract_text_using_pdfminer(pdf_path):
    """Extract text using PDFMiner"""
    return pdfminer.high_level.extract_text(str(pdf_path))

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

# Move EXTRACTION_METHODS after all extraction functions
EXTRACTION_METHODS = [
    ('MarkItDown', extract_text_using_markitdown),
    ('PDFMiner', extract_text_using_pdfminer),
    ('PyPDF', fallback_pdf_extraction)
]

# Improve API calls with better error handling
def make_api_call(messages, filename):
    """Make API call with retry logic and error handling"""
    try:
        response = client.chat.completions.create(
            model=API_CONFIG['model'],
            messages=messages,
            max_tokens=API_CONFIG['max_tokens'],
            temperature=API_CONFIG['temperature']
        )
        
        if not response.choices:
            raise ValueError("No response choices available")
            
        return (
            response.choices[0].message.content.strip() if response.choices[0].message.content else "",
            response.usage.total_tokens if response.usage else 0
        )
    except Exception as e:
        logger.error(f"[{filename}] API call failed: {str(e)}")
        return "", 0

def extract_title_from_text(cleaned_text, fallback_name):
    """Extract paper title from cleaned text using DeepSeek"""
    messages = [
        {
            "role": "system",
            "content": TITLE_EXTRACTION_PROMPT
        },
        {
            "role": "user",
            "content": cleaned_text[:1000]
        }
    ]
    
    title, tokens = make_api_call(messages, fallback_name)
    return (title if title else fallback_name), tokens

def clean_text_with_deepseek(text, filename):
    """Use DeepSeek to clean and structure the extracted text"""
    total_tokens = 0
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def process_chunk(chunk, is_continuation=False):
        nonlocal total_tokens
        messages = [
            {
                "role": "system",
                "content": CONTINUATION_PROMPT if is_continuation else TEXT_CLEANING_PROMPT
            },
            {
                "role": "user",
                "content": chunk
            }
        ]
        result, tokens = make_api_call(messages, filename)
        total_tokens += tokens
        return result

    try:
        if len(text) > FILE_PROCESSING['max_chunk_size']:
            chunks = [
                text[i:i + FILE_PROCESSING['max_chunk_size']] 
                for i in range(0, len(text), FILE_PROCESSING['max_chunk_size'])
            ]
            cleaned_chunks = []
            
            logger.info(f"[{filename}] Processing {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks, 1):
                if interrupted:
                    return clean_text(text), total_tokens
                
                context = (cleaned_chunks[-1][-FILE_PROCESSING['overlap_size']:] 
                          if cleaned_chunks else "")
                result = process_chunk(
                    context + chunk if cleaned_chunks else chunk,
                    is_continuation=bool(cleaned_chunks)
                )
                cleaned_chunks.append(result)
                logger.info(f"[{filename}] Processed chunk {i}/{len(chunks)}")
            
            return "\n\n".join(cleaned_chunks), total_tokens
        else:
            result = process_chunk(text)
            return result, total_tokens

    except Exception as e:
        logger.error(f"[{filename}] DeepSeek cleaning failed: {e}")
        return clean_text(text), total_tokens

# Improve PDF processing with better error handling
def process_pdf(pdf_path, output_dir):
    """Process a single PDF file"""
    if interrupted:
        return False
        
    filename = pdf_path.name
    logger.info(f"Processing: {filename}")
    
    try:
        # Extract text using available methods
        text_content = None
        for method_name, extractor in EXTRACTION_METHODS:
            try:
                text_content = extractor(pdf_path)
                if text_content and text_content.strip():
                    logger.debug(f"[{filename}] Extracted text using {method_name}")
                    break
            except Exception as e:
                logger.debug(f"[{filename}] {method_name} failed: {e}")
        
        if not text_content:
            logger.warning(f"[{filename}] No text could be extracted")
            return False

        # Process and save the text
        logger.info(f"[{filename}] Cleaning text...")
        cleaned_text, cleaning_tokens = clean_text_with_deepseek(text_content, filename)
        
        if interrupted:
            return False
            
        logger.info(f"[{filename}] Extracting title...")
        paper_title, title_tokens = extract_title_from_text(cleaned_text, pdf_path.stem)
        
        output_file = Path(output_dir) / f"{clean_filename(paper_title)}.md"
        if output_file.exists():
            logger.info(f"[{filename}] Skipping - Output file already exists")
            return False
        
        output_file.write_text(cleaned_text, encoding='utf-8')
        
        logger.info(
            f"[{filename}] Converted → {output_file.name} "
            f"(Tokens: {cleaning_tokens + title_tokens:,})"
        )
        return True
            
    except Exception as e:
        logger.error(f"[{filename}] Failed to process: {str(e)}")
        return False

# Main conversion function with improved parallel processing
def convert_pdfs_to_md(input_dir: str, output_dir: str, compress_md: bool = False):
    """Convert PDFs to Markdown with parallel processing and proper cleanup"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return

    max_workers = min(FILE_PROCESSING['max_workers'], len(pdf_files))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_pdf, pdf_path, output_dir): pdf_path 
            for pdf_path in pdf_files
        }
        
        stats = {'converted': 0, 'skipped': 0}
        
        try:
            for future in as_completed(futures):
                if interrupted:
                    break
                if future.result():
                    stats['converted'] += 1
                else:
                    stats['skipped'] += 1
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            logger.info("\nOperation interrupted by user")
            return
        finally:
            logger.info("\nSummary:")
            logger.info(f"Total files: {len(pdf_files)}")
            logger.info(f"Converted: {stats['converted']}")
            logger.info(f"Skipped: {stats['skipped']}")

# Main entry point
def main():
    parser = argparse.ArgumentParser(description='PDF to Markdown Converter')
    parser.add_argument('input_dir', help='Input PDF directory')
    parser.add_argument('output_dir', help='Output Markdown directory')
    args = parser.parse_args()
    convert_pdfs_to_md(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()