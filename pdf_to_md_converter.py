# Standard library imports
import os
import re
import sys
import time
import queue
import signal
import logging
import warnings
import argparse
import threading
import unicodedata
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import pypdf
import openai
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from markitdown import MarkItDown
import pdfminer.high_level
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.utils import CryptographyDeprecationWarning

# Constants
API_CONFIG = {
    'deepseek': {
        'model': 'deepseek-chat',
        'max_tokens': 4000,
        'temperature': 0.1,
        'base_url': 'https://api.deepseek.com/v1'
    },
    'xai': {
        'model': 'grok-beta',
        'max_tokens': 4000,
        'temperature': 0.1,
        'base_url': 'https://api.x.ai/v1'
    }
}

FILE_PROCESSING = {
    'max_chunk_size': 12000,
    'max_filename_length': 255,
    'max_workers': 32,
    'overlap_size': 200
}

PROMPTS = {
    'title_extraction': """You are a precise academic paper title extractor. 
Extract only the main title from the text, without any additional content. 
The text is already cleaned and formatted.""",

    'text_cleaning': """You are a text cleaner specialized in academic papers. Clean and structure the text while:

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

Output ONLY the cleaned text without explanations.""",

    'continuation': """Continue cleaning the text while preserving all markdown formatting. 
Output ONLY the cleaned text without any explanations.""",

    'classification': """You are a document classifier. Classify academic documents as one of:
- journal_paper
- conference_paper
- book_chapter
- thesis
- technical_report
- preprint
- unknown

Return only the classification label, nothing else."""
}

# Global state management
class State:
    """Manages global application state"""
    def __init__(self):
        self.interrupted = False
        self.client: openai.OpenAI | None = None

state = State()

# Configure logging
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

# Suppress warnings and logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# Move API client configuration to a separate function
def configure_api_client(provider='deepseek'):
    """Configure and return API client for specified provider"""
    if provider == 'deepseek':
        return openai.OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=API_CONFIG['deepseek']['base_url']
        )
    elif provider == 'xai':
        return openai.OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url=API_CONFIG['xai']['base_url'],
            default_headers={
                "X-API-Version": "1.0",
                "Content-Type": "application/json"
            }
        )
    else:
        raise ValueError(f"Unsupported API provider: {provider}")
        return None

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    if not state.interrupted:
        logger.info("\nGracefully shutting down... (Press Ctrl+C again to force quit)")
        state.interrupted = True
    else:
        logger.info("\nForce quitting...")
        sys.exit(1)

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add back the missing text cleaning functions that were referenced earlier
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
def make_api_call(messages, filename, provider='deepseek'):
    """Make API call with retry logic and error handling"""
    try:
        config = API_CONFIG[provider]
        
        if provider == 'xai':
            formatted_messages = [
                {
                    "role": m["role"],
                    "content": m["content"].replace("\n\n", "\n").strip()
                }
                for m in messages
            ]
        else:
            formatted_messages = messages

        if not state.client:
            raise ValueError("API client not configured")
        
        response = state.client.chat.completions.create(
            model=config['model'],
            messages=[{"role": m["role"], "content": m["content"]} for m in formatted_messages],
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.choices:
            raise ValueError("No response choices available")
            
        return (
            response.choices[0].message.content.strip() if response.choices[0].message.content else "",
            response.usage.total_tokens if response.usage else 0
        )
    except Exception as e:
        logger.error(f"[{filename}] API call failed ({provider}): {str(e)}")
        return "", 0

def extract_title_from_text(cleaned_text, fallback_name, provider='deepseek'):
    """Extract paper title from cleaned text using AI"""
    messages = [
        {
            "role": "system",
            "content": PROMPTS['title_extraction']
        },
        {
            "role": "user",
            "content": cleaned_text[:1000]
        }
    ]
    
    title, tokens = make_api_call(messages, fallback_name, provider)
    return (title if title else fallback_name), tokens

def clean_text_with_ai(text, filename, provider='deepseek'):
    """Use AI to clean and structure the extracted text"""
    total_tokens = 0
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def process_chunk(chunk, is_continuation=False):
        nonlocal total_tokens
        messages = [
            {
                "role": "system",
                "content": PROMPTS['continuation'] if is_continuation else PROMPTS['text_cleaning']
            },
            {
                "role": "user",
                "content": chunk
            }
        ]
        result, tokens = make_api_call(messages, filename, provider)
        total_tokens += tokens
        return result

    try:
        # Adjust chunk size based on provider
        max_chunk_size = (
            FILE_PROCESSING['max_chunk_size'] 
            if provider == 'deepseek' 
            else FILE_PROCESSING['max_chunk_size'] // 2  # XAI typically prefers smaller chunks
        )

        if len(text) > max_chunk_size:
            chunks = [
                text[i:i + max_chunk_size] 
                for i in range(0, len(text), max_chunk_size)
            ]
            cleaned_chunks = []
            
            logger.info(f"[{filename}] Processing {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks, 1):
                if state.interrupted:
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
        logger.error(f"[{filename}] AI cleaning failed ({provider}): {e}")
        return clean_text(text), total_tokens

# Improve PDF processing with better error handling
def process_pdf(pdf_path, output_dir, provider='deepseek'):
    """Process a single PDF file"""
    if state.interrupted:
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
        cleaned_text, cleaning_tokens = clean_text_with_ai(text_content, filename, provider)
        
        if state.interrupted:
            return False
            
        logger.info(f"[{filename}] Extracting title...")
        paper_title, title_tokens = extract_title_from_text(cleaned_text, pdf_path.stem, provider)
        
        # Save the markdown
        save_optimized_markdown(cleaned_text, output_dir, paper_title, provider)
        
        logger.info(
            f"[{filename}] Converted → {clean_filename(paper_title)}.md "
            f"(Tokens: {cleaning_tokens + title_tokens:,})"
        )
        return True
            
    except Exception as e:
        logger.error(f"[{filename}] Failed to process: {str(e)}")
        return False

# Add a new class to manage the processing queue
class PDFProcessor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.queue = queue.Queue()
        self.stats = {'converted': 0, 'skipped': 0}
        self.lock = threading.Lock()

    def process_queue(self):
        while True:
            try:
                if state.interrupted:
                    break
                    
                pdf_path, output_dir, provider = self.queue.get_nowait()
                success = process_pdf(pdf_path, output_dir, provider)
                
                with self.lock:
                    if success:
                        self.stats['converted'] += 1
                    else:
                        self.stats['skipped'] += 1
                        
                self.queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                break

def convert_pdfs_to_md(input_dir: str, output_dir: str, provider: str = 'deepseek'):
    """Convert PDFs to Markdown with specified AI provider"""
    client = configure_api_client(provider)
    if not client:
        logger.error(f"Failed to configure API client for {provider}")
        return
    
    state.client = client
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.warning(f"No PDFs found in {input_dir}")
        return

    max_workers = min(FILE_PROCESSING['max_workers'], len(pdf_files))
    processor = PDFProcessor(max_workers)
    
    # Fill the queue
    for pdf_path in pdf_files:
        processor.queue.put((pdf_path, output_dir, provider))

    # Create and start worker threads
    workers = []
    for _ in range(max_workers):
        thread = threading.Thread(target=processor.process_queue)
        thread.daemon = True
        workers.append(thread)
        thread.start()

    try:
        # Wait for all tasks to complete or interruption
        while any(thread.is_alive() for thread in workers):
            if state.interrupted:
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        state.interrupted = True
        logger.info("\nOperation interrupted by user")

    finally:
        # Wait briefly for threads to clean up
        for thread in workers:
            thread.join(timeout=1.0)
            
        logger.info("\nSummary:")
        logger.info(f"Total files: {len(pdf_files)}")
        logger.info(f"Converted: {processor.stats['converted']}")
        logger.info(f"Skipped: {processor.stats['skipped']}")

def optimize_for_llm_context(text, max_chunk_size=8000):
    """Split text into LLM-friendly chunks while preserving structure"""
    # Preserve markdown headers as chunk boundaries
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split by headers first
    sections = re.split(r'(^#{1,6}\s.*$)', text, flags=re.MULTILINE)
    
    for section in sections:
        section_size = len(section)
        
        if current_size + section_size > max_chunk_size:
            # Save current chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size
    
    # Add remaining content
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def save_optimized_markdown(text, output_dir, title, provider='deepseek'):
    """Save markdown in LLM-optimized chunks"""
    chunks = optimize_for_llm_context(text)
    output_dir = Path(output_dir)
    safe_title = clean_filename(title)
    
    # Detect paper type once for consistency
    paper_type = detect_paper_type(text, title, 'untitled', provider)
    
    if len(chunks) == 1:
        # Single file if content fits context window
        content = add_frontmatter(chunks[0], title, paper_type=paper_type)
        (output_dir / f"{safe_title}.md").write_text(content, encoding='utf-8')
    else:
        # Create directory for split content
        paper_dir = output_dir / safe_title
        paper_dir.mkdir(exist_ok=True)
        
        # Create index file
        index_content = add_frontmatter(
            f"# {title}\n\n## Contents\n\n",
            title,
            paper_type=paper_type,
            is_index=True
        )
        
        # Save chunks with proper linking
        for i, chunk in enumerate(chunks, 1):
            chunk_file = f"part_{i:02d}.md"
            # Add frontmatter to each chunk with part number
            content = add_frontmatter(
                chunk,
                title,
                chunk_number=i,
                paper_type=paper_type
            )
            (paper_dir / chunk_file).write_text(content, encoding='utf-8')
            index_content += f"- [Part {i}]({chunk_file})\n"
        
        # Save index
        (paper_dir / "index.md").write_text(index_content, encoding='utf-8')

def add_frontmatter(text, title, chunk_number=None, paper_type=None, is_index=False):
    """Add YAML frontmatter for better indexing
    
    Args:
        text: The document text
        title: Document title
        chunk_number: Part number for multi-part documents
        paper_type: Type of academic document
        is_index: Whether this is an index file
    """
    frontmatter = [
        "---",
        f"title: {title}",
        f"type: {paper_type or 'unknown'}"
    ]
    
    if is_index:
        frontmatter.extend([
            "part: index"
        ])
    else:
        frontmatter.extend([
            f"part: {chunk_number if chunk_number else 'index'}"
        ])
    
    frontmatter.append("---\n")
    return "\n".join(frontmatter) + text

def detect_paper_type(text, title, filename, provider='deepseek'):
    """Use LLM to detect academic document type"""
    sample = text[:1000]
    
    messages = [
        {
            "role": "system",
            "content": PROMPTS['classification']
        },
        {
            "role": "user",
            "content": f"Title: {title}\n\nExcerpt:\n{sample}"
        }
    ]
    
    result, _ = make_api_call(messages, filename, provider)
    return result.lower().strip() or 'unknown'

# Main entry point
def main():
    parser = argparse.ArgumentParser(description='PDF to Markdown Converter')
    parser.add_argument('input_dir', help='Input PDF directory')
    parser.add_argument('output_dir', help='Output Markdown directory')
    parser.add_argument(
        '--provider', 
        choices=['deepseek', 'xai'], 
        default='deepseek',
        help='AI provider to use (default: deepseek)'
    )
    args = parser.parse_args()
    convert_pdfs_to_md(args.input_dir, args.output_dir, args.provider)

if __name__ == '__main__':
    main()