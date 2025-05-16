"""
Module for processing academic papers and extracting citations.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from bs4 import BeautifulSoup
import json
from pathlib import Path
import re

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

    def get_masked_text_and_citations(self, para, bib_entries):
        masked_parts = []
        citation_keys = set()
        
        para_copy = BeautifulSoup(str(para), features="html.parser")
        
        for cite in para_copy.find_all("cite"):
            refs = cite.find_all("a", class_="ltx_ref")
            for ref in refs:
                href = ref.get("href", "")
                match = re.search(r"#(bib\.bib\d+)", href)
                if match:
                    key = match.group(1)
                    if key in bib_entries:
                        citation_keys.add(key)
            cite.replace_with(" [CITATION]")
        
        for elem in para_copy.descendants:
            if isinstance(elem, str):
                masked_parts.append(elem)
        
        masked_text = re.sub(r"\s+", " ", "".join(masked_parts)).strip()
        return masked_text, citation_keys

    def extract_citation_info_from_html(self, html_path) -> dict:
        """
        Extracts citation information from a given HTML file.
        Args:
            html_path (str): Path to the HTML file.    
        Returns:
            dict: A dictionary containing citation information, where the keys are entry IDs and the values are dictionaries with citation details.
        """

        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        biblist = soup.find("ul", class_="ltx_biblist")
        bib_entries = {}

        for li in biblist.find_all("li", class_="ltx_bibitem"):
            entry_id = li.get("id", "")
            
            tag_span = li.find("span", class_="ltx_tag ltx_role_refnum ltx_tag_bibitem")
            tag = tag_span.get_text(strip=True) if tag_span else ""
            if tag[-1] == '↑': # Remove button text
                tag = tag[:-1]

            bibblocks = li.find_all("span", class_="ltx_bibblock")
            authors = bibblocks[0].get_text(strip=True) if bibblocks else ""
            
            title = bibblocks[1].get_text(strip=True) if bibblocks and len(bibblocks) > 1 else ""
            if title == "":
                a_tag = li.find("a", class_="ltx_ref")
                title = a_tag.get_text(strip=True) if a_tag else ""

            journal = bibblocks[2].get_text(strip=True) if bibblocks and len(bibblocks) > 2 else ""

            bib_entries[entry_id] = {
                "tag": tag,
                "title": title,
                "authors": authors,
                'journal': journal
            }
        return bib_entries
        
    def process_html(self, html_path: Union[str, Path]) -> Dict[str, Any]:
        html_path = Path(html_path)
        logger.info(f"Processing HTML: {html_path}")
        
        bib_entries = self.extract_citation_info_from_html(html_path)

        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        full_text = []
        full_masked_text = []
        full_citation_keys = set()
        sections_dict = {}
        paragraphs_dict = {}


        for section in soup.find_all("section", class_="ltx_section"):
            section_id = section.get("id", "")
            section_title_tag = section.find("h2", class_="ltx_title")
            section_title = section_title_tag.get_text(strip=True) if section_title_tag else section_id

            section_citations_keys = set()
            section_masked_text = []
            section_text = []

            for para in section.find_all("div", class_="ltx_para"):
                para_id = para.get("id", "")
                para_text = para.get_text(separator=" ", strip=True)
                
                masked_text, citations_keys = self.get_masked_text_and_citations(para, bib_entries)

                citations = [bib_entries[key] for key in citations_keys if key in bib_entries]

                paragraphs_dict[para_id] = {
                    "text": para_text,
                    "masked_text": masked_text,
                    "citations": citations
                }

                section_masked_text.append(masked_text)
                full_masked_text.append(masked_text)
                
                section_text.append(para_text)
                full_text.append(para_text)

                section_citations_keys.update(citations_keys)

            section_citations = [bib_entries[key] for key in section_citations_keys if key in bib_entries]
            sections_dict[section_id] = {
                "title": section_title,
                "text": "\n".join(section_text),
                "masked_text": "\n".join(section_masked_text),
                "citations": section_citations
            }

            full_citation_keys.update(section_citations_keys)


        full_citations = [bib_entries[key] for key in full_citation_keys if key in bib_entries]

        full_dict = {
            "title": soup.find("h1", class_="ltx_title ltx_title_document").get_text(strip=True) if soup.find("h1", class_="ltx_title ltx_title_document") else "Untitled",
            "text": "\n".join(full_text),
            "masked_text": "\n".join(full_masked_text),
            "citations": full_citations
        }


        parsed_result = {
            "title": full_dict["title"],
            "full": full_dict,
            "sections": sections_dict,
            "paragraphs": paragraphs_dict
        }
            
        # Save the processed results
        self._save_processed_data(parsed_result, html_path.stem)
        
        return parsed_result
    
    def process_directory(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        
        input_dir = Path(input_dir)
        logger.info(f"Processing all HTMLs in directory: {input_dir}")
        
        results = []
        html_files = list(input_dir.glob("*.html"))
        
        for file in tqdm(html_files, desc="Processing papers"):
            try:
                result = self.process_html(file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                exit(0)
        
        return results
    
    def _save_processed_data(self, data: Dict[str, Any], base_filename: str) -> None:
        """
        Save processed data to output files.
        
        Args:
            data: Dictionary containing processed data
            base_filename: Base filename to use for output files
        """
        # Create output subdirectories
        if len(base_filename) > 20:
            base_filename = base_filename[:20]

        if base_filename[-1] == '.':
            base_filename = base_filename[:-1]
        
        full_paper_dir = self.output_dir / "full_paper"
        section_dir = self.output_dir / "section"
        paragraph_dir = self.output_dir / "paragraph"
        
        full_paper_dir.mkdir(exist_ok=True)
        section_dir.mkdir(exist_ok=True)
        paragraph_dir.mkdir(exist_ok=True)
        
        # Save files
        with open(full_paper_dir / f"{base_filename}.json", "w", encoding="utf-8") as f:
            json.dump(data["full"], f, indent=2)
            
        with open(section_dir / f"{base_filename}.json", "w", encoding="utf-8") as f:
            json.dump(data["sections"], f, indent=2)
            
        with open(paragraph_dir / f"{base_filename}.json", "w", encoding="utf-8") as f:
            json.dump(data["paragraphs"], f, indent=2)


if __name__ == "__main__":
    # Example usage
    processor = PaperProcessor(output_dir="data/output/processed_ACL_html")
    # results = processor.process_pdf("data\input\mini_test\GENMO.txt")
    results = processor.process_directory(r"data\input\ACL_papers\html")
    # print(json.dumps(results, indent=2))