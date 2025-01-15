import os
import re
import time
import logging
import warnings
import argparse
import unicodedata
from pathlib import Path

# Third-party imports
import spacy
import pypdf
import requests
import camelot.io as camelot
from tqdm import tqdm
from dotenv import load_dotenv
from markitdown import MarkItDown
import pdfminer.high_level
from pdfminer.layout import LAParams, LTTextContainer
from pdfminer.high_level import extract_pages

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

# PDF extraction functions
def extract_text_with_layout(pdf_path):
    """Extract text while preserving layout and formatting"""
    text_content = []
    for page_layout in extract_pages(pdf_path, laparams=LAParams()):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text_content.append(element.get_text())
    return '\n'.join(text_content)

def extract_tables(pdf_path):
    """Extract tables from PDF using Camelot"""
    tables = camelot.read_pdf(pdf_path)
    markdown_tables = []
    for table in tables:
        markdown = "| " + " | ".join(table.df.columns) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(table.df.columns)) + " |\n"
        for _, row in table.df.iterrows():
            markdown += "| " + " | ".join(row) + " |\n"
        markdown_tables.append(markdown)
    return markdown_tables

def extract_math_formulas(text):
    """Extract mathematical formulas using regex patterns"""
    formula_patterns = [
        r'\$(.*?)\$',  # Inline math
        r'\[\[(.*?)\]\]',  # Display math
    ]
    formulas = []
    for pattern in formula_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            formulas.append(match.group(1))
    return formulas

# NLP functions
def detect_sections(text):
    """Improved section detection using NLP"""
    nlp = spacy.load("en_core_web_sm")
    section_headers = {
        'abstract': ['abstract', 'summary'],
        'introduction': ['introduction', 'background'],
        'methodology': ['methodology', 'methods', 'approach'],
        'results': ['results', 'findings', 'evaluation'],
        'conclusion': ['conclusion', 'conclusions', 'future work'],
        'references': ['references', 'bibliography']
    }
    doc = nlp(text)
    sections = {}
    for section, keywords in section_headers.items():
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in keywords):
                sections[section] = sent.start_char
    return sections

# Conversion functions
def convert_pdf(pdf_path):
    """Convert a single PDF file to markdown format"""
    try:
        text_content = extract_text_with_layout(pdf_path)
        sections = detect_sections(text_content)
        tables = extract_tables(pdf_path)
        formulas = extract_math_formulas(text_content)
        
        markdown_content = text_content
        
        if tables:
            markdown_content += "\n\n## Tables\n\n" + "\n\n".join(tables)
        if formulas:
            markdown_content += "\n\n## Mathematical Formulas\n\n"
            for formula in formulas:
                markdown_content += f"\n${formula}$\n"
        
        return markdown_content
    except Exception as e:
        logger.error(f"Error converting {pdf_path}: {e}")
        raise

def safe_convert(pdf_path, retries=3):
    """Conversion with error recovery"""
    for attempt in range(retries):
        try:
            return convert_pdf(pdf_path)
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to convert {pdf_path} after {retries} attempts: {e}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)

def convert_batch(pdf_files):
    """Convert multiple PDFs with progress bar"""
    results = []
    with tqdm(total=len(pdf_files), desc="Converting PDFs") as pbar:
        for pdf in pdf_files:
            results.append(convert_pdf(pdf))
            pbar.update(1)
    return results

def convert_pdfs_to_md(input_dir: str, output_dir: str):
    """Convert PDF files to Markdown, skipping already converted files"""
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
            
            cleaned_text = clean_text(text_content)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
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