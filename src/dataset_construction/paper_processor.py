"""
Module for processing academic papers and extracting citations.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from ..utils import extract_citations

import PyPDF2
import pdfplumber
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PaperProcessor:
    """
    Class for processing academic papers, extracting citations, and generating
    different versions of the paper with varying levels of citation information.
    """
    
    # Common citation patterns in academic papers

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the PaperProcessor.
        
        Args:
            output_dir: Directory to save processed output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF file to extract text and citations.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing processed data
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        if pdf_path.suffix.lower() == ".txt":
            with open(pdf_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            text = self._extract_text_from_pdf(pdf_path)
        
        # Extract citations and their positions
        citations, citation_positions = self._extract_citations(text)
        
        # Generate different versions of the paper
        no_citations_text = self._remove_citations(text, citations)
        marked_citations_text = self._mark_citations(no_citations_text, citations, citation_positions)
        
        # Prepare output
        result = {
            "original_filename": pdf_path.name,
            "original_text": text,
            "no_citations_text": no_citations_text,
            "marked_citations_text": marked_citations_text,
            "citations": {
                "content": citations,
                "positions": citation_positions
            }
        }
        
        # Save the processed results
        self._save_processed_data(result, pdf_path.stem)
        
        return result
    
    def process_directory(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            
        Returns:
            List of dictionaries containing processed data for each PDF
        """
        input_dir = Path(input_dir)
        logger.info(f"Processing all PDFs in directory: {input_dir}")
        
        results = []
        pdf_files = list(input_dir.glob("*.pdf"))
        txt_files = list(input_dir.glob("*.txt"))
        pdf_files.extend(txt_files)
        
        for pdf_file in tqdm(pdf_files, desc="Processing papers"):
            try:
                result = self.process_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        return results
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text with pdfplumber: {str(e)}")
            
            # Fallback to PyPDF2
            try:
                text = ""
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Failed to extract text with PyPDF2: {str(e)}")
                raise
    
    def _extract_citations(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            Tuple containing list of citations and their positions
        """
        # citations = []
        # positions = []
        
        # # Apply all citation patterns
        # for pattern in self.CITATION_PATTERNS:
        #     for match in re.finditer(pattern, text):
        #         citation = match.group(0)
        #         start, end = match.span()
                
        #         citations.append(citation)
        #         positions.append((start, end))
        
        # # Sort by position
        # sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i][0])
        # sorted_citations = [citations[i] for i in sorted_indices]
        # sorted_positions = [positions[i] for i in sorted_indices]
        
        sorted_citations, sorted_positions = extract_citations(text)
        
        return sorted_citations, sorted_positions
    
    def _remove_citations(self, text: str, citations: List[str]) -> str:
        """
        Remove all citations from text.
        
        If '++ref++' is detected, remove everything between '++ref++[' and ']++ref++'.
        Otherwise, remove citation strings from the provided list.
        
        Args:
            text: Original text
            citations: List of citations to remove
        
        Returns:
            Text with citations removed
        """
        result = text
        if "++ref++" in result:
            # Remove all ++ref++[ ... ]++ref++ blocks
            # result = re.sub(r'\+\+ref\+\+\[\s*(?:\((?:[^()]+)\)\s*,?\s*)*\]\+\+ref\+\+', '', result)
            # result = re.sub(r'\+\+ref\+\+\[[\s\S]*?\]\+\+ref\+\+', '', result)
            result = re.sub(r'\+\+ref\+\+\[ [\s\S]*? \]\+\+ref\+\+', '', result)
            return result
        
        # Remove any additional citations provided
        for citation in citations:
            result = result.replace(citation, "")
        
        return result
    
    def _mark_citations(self, no_citations_text: str, citations: List[str], 
                       positions: List[int]) -> str:
        """
        Mark citation positions but remove their content.
        
        Args:
            no_citations_text: No-citation text
            citations: List of citations
            positions: List of citation positions start position. 
            
        Returns:
            Text with citation positions marked
        """
        ctn = 0
        result = copy(no_citations_text)
        accumulated_offset = 0
        
        for start in positions:
            mark = f"[CITATION_{ctn}]"
            mark_length = len(mark)
            ctn += 1
            
            accumulated_start = start + accumulated_offset
            result = result[:accumulated_start] + mark + result[accumulated_start:]
            accumulated_offset += mark_length
        
        return result
    
    def _save_processed_data(self, data: Dict[str, Any], base_filename: str) -> None:
        """
        Save processed data to output files.
        
        Args:
            data: Dictionary containing processed data
            base_filename: Base filename to use for output files
        """
        # Create output subdirectories
        no_citations_dir = self.output_dir / "no_citations"
        marked_citations_dir = self.output_dir / "marked_citations"
        citations_dir = self.output_dir / "citations"
        
        no_citations_dir.mkdir(exist_ok=True)
        marked_citations_dir.mkdir(exist_ok=True)
        citations_dir.mkdir(exist_ok=True)
        
        # Save files
        with open(no_citations_dir / f"{base_filename}.txt", "w", encoding="utf-8") as f:
            f.write(data["no_citations_text"])
            
        with open(marked_citations_dir / f"{base_filename}.txt", "w", encoding="utf-8") as f:
            f.write(data["marked_citations_text"])
            
        with open(citations_dir / f"{base_filename}.json", "w", encoding="utf-8") as f:
            json.dump(data["citations"], f, indent=2)


if __name__ == "__main__":
    # Example usage
    processor = PaperProcessor(output_dir="data/output/processed")
    # results = processor.process_pdf("data\input\mini_test\GENMO.txt")
    results = processor.process_directory("data/input/mini_test")
    print(json.dumps(results, indent=2))