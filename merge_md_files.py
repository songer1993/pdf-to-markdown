import os
import argparse
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('md_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def merge_markdown_files(input_dir: str, output_file: str, add_headers: bool = True):
    """
    Merge all markdown files in the input directory into a single file
    
    Args:
        input_dir (str): Directory containing markdown files
        output_file (str): Path to the output merged file
        add_headers (bool): Whether to add file names as headers
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Get all .md files
    md_files = sorted(input_path.glob('*.md'))
    
    if not md_files:
        logger.warning(f"No markdown files found in {input_dir}")
        return
    
    logger.info(f"Found {len(md_files)} markdown files")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, md_file in enumerate(md_files, 1):
            try:
                with open(md_file, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    
                    # Add a newline before content if not the first file
                    if i > 1:
                        outfile.write('\n\n')
                    
                    # Add filename as header if requested
                    if add_headers:
                        header = f"# {md_file.stem}"
                        outfile.write(f"{header}\n\n")
                    
                    outfile.write(content)
                
                logger.info(f"✓ Merged: {md_file.name}")
                
            except Exception as e:
                logger.error(f"✗ Error merging {md_file.name}: {e}")
    
    logger.info(f"\nMerged files saved to: {output_path}")

def split_markdown_file(input_file: str, chunk_size_kb: int = 200):
    """
    Split a large markdown file into smaller chunks
    
    Args:
        input_file (str): Path to the input markdown file
        chunk_size_kb (int): Maximum size of each chunk in KB
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Read the entire file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into sections (by headers)
    sections = re.split(r'(?=# )', content.strip())
    sections = [s for s in sections if s.strip()]  # Remove empty sections
    
    current_chunk = []
    current_size = 0
    chunk_num = 1
    
    for section in sections:
        section_size = len(section.encode('utf-8')) / 1024  # Size in KB
        
        # If adding this section would exceed chunk size, write current chunk
        if current_size + section_size > chunk_size_kb and current_chunk:
            # Write current chunk
            chunk_path = input_path.parent / f"{input_path.stem}_chunk{chunk_num}{input_path.suffix}"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(''.join(current_chunk))
            logger.info(f"Created chunk {chunk_num}: {chunk_path}")
            
            # Start new chunk
            chunk_num += 1
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size
    
    # Write remaining content
    if current_chunk:
        chunk_path = input_path.parent / f"{input_path.stem}_chunk{chunk_num}{input_path.suffix}"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(''.join(current_chunk))
        logger.info(f"Created chunk {chunk_num}: {chunk_path}")
    
    logger.info(f"\nSplit {input_path.name} into {chunk_num} chunks")

def main():
    parser = argparse.ArgumentParser(description='Markdown File Merger and Splitter')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge markdown files')
    merge_parser.add_argument('input_dir', help='Input directory containing markdown files')
    merge_parser.add_argument('output_file', help='Output merged markdown file')
    merge_parser.add_argument('--no-headers', action='store_true', 
                            help='Do not add file names as headers')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split markdown file into chunks')
    split_parser.add_argument('input_file', help='Input markdown file to split')
    split_parser.add_argument('--chunk-size', type=int, default=200,
                            help='Maximum size of each chunk in KB (default: 200)')
    
    args = parser.parse_args()
    
    if args.command == 'merge':
        merge_markdown_files(args.input_dir, args.output_file, not args.no_headers)
    elif args.command == 'split':
        split_markdown_file(args.input_file, args.chunk_size)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 