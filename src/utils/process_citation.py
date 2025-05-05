from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import re

CITATION_PATTERNS = [
        r'\+\+ref\+\+\[[\s\S]*?\]\+\+ref\+\+',
        r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]',  # [1] or [1,2,3]
        r'\[\s*[A-Za-z]+\s*(?:et al\.)?\s*,?\s*\d{4}(?:[a-z])?\s*\]',  # [Smith et al., 2020]
        r'\(\s*[A-Za-z]+\s*(?:et al\.)?\s*,?\s*\d{4}(?:[a-z])?\s*\)',  # (Smith et al., 2020)
        r'\(\s*[A-Za-z]+\s*and\s*[A-Za-z]+\s*,?\s*\d{4}(?:[a-z])?\s*\)',  # (Smith and Jones, 2020)
        r'\[\s*[A-Za-z]+\s*and\s*[A-Za-z]+\s*,?\s*\d{4}(?:[a-z])?\s*\]',  # [Smith and Jones, 2020]
    ]

def parse_tokenized_titles(citation_text):
    """
    Parse a special tokenized citation block like:
    ++ref++[ (Title1), (Title2), ... ]++ref++
    and return a list of clean titles.
    """
    pattern = r'\+\+ref\+\+\[([\s\S]*?)\]\+\+ref\+\+'
    match = re.match(pattern, citation_text)
    if not match:
        return []

    content = match.group(1)
    # Extract individual titles wrapped in parentheses
    titles = re.findall(r'\(([^()]+)\)', content)
    # Strip whitespace from each title
    return [title.strip() for title in titles]

def extract_citations(text: str) -> Tuple[List[str], List[int]]:
        """
        Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            Tuple containing list of citations and their positions
            positions: is the position in the no-citation text
        """
        citations = []
        positions = []
        
        # Apply all citation patterns
        for pattern in CITATION_PATTERNS:
            for match in re.finditer(pattern, text):
                citation = match.group(0)
                start, end = match.span()
                
                if "++ref++" in citation:
                    # Handle special case for ++ref++ citations
                    citation = parse_tokenized_titles(citation)
                    
                
                citations.append(citation)
                positions.append((start, end))
        
        # Sort by position
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i][0])
        sorted_citations = [citations[i] for i in sorted_indices]
        sorted_positions = [positions[i] for i in sorted_indices]
        
        # Calculate positions in no-citation text
        no_citation_positions = []
        offset = 0
        
        for i, (start, end) in enumerate(sorted_positions):
            # Adjust for previous removals
            adjusted_start = start - offset
            citation_length = end - start
            
            # Record position in no-citation text
            no_citation_positions.append(adjusted_start)
            
            # Update offset for future positions
            offset += citation_length
        
        return sorted_citations, no_citation_positions
    
    
if __name__ == "__main__":
    text = "citations [1], [2,3], and (Smith et al., 2020)."
    clean_text = "citations , , and ."
    citations, positions = extract_citations(text)
    
    print("Citations:", citations)
    print("Positions:", positions)
    
    for citation, pos in zip(citations, positions):
        start = pos
        context = clean_text[start : start + 1]
        print(f"Citation: {citation} at position {pos}")
        print(f"Context: \"{context}\"")