"""
Module providing baseline citation prediction methods.
"""

import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RandomCitationPredictor:
    """
    Random baseline citation predictor.
    Randomly inserts citations at plausible positions in the text.
    
    This serves as a simple baseline to compare more sophisticated methods against.
    """
    
    def __init__(self, citations_corpus_path: Optional[Union[str, Path]] = None):
        """
        Initialize the RandomCitationPredictor.
        
        Args:
            citations_corpus_path: Path to JSON file containing corpus of citations to sample from
        """
        self.citations_corpus = self._load_citations_corpus(citations_corpus_path)
        
        # Patterns for identifying potential citation locations
        self.citation_location_patterns = [
            r'\b(?:according to|as shown by|as demonstrated by|as reported by)\b',
            r'\b(?:previous work|prior work|related work|previous studies|prior studies|related studies)\b',
            r'\b(?:recently|previously|earlier studies|earlier work)\b',
            r'\b(?:has been proposed|has been shown|has been demonstrated|has been suggested)\b',
            r'(?<=[.!?])\s+[A-Z]',  # Start of a sentence (potential citation at end of previous sentence)
            r'\s+(?:shows|demonstrates|suggests|indicates|proposes)\s+that\b',
        ]
        
        # Combine patterns
        self.combined_pattern = '|'.join(self.citation_location_patterns)
    
    def _load_citations_corpus(self, path: Optional[Union[str, Path]]) -> List[str]:
        """
        Load corpus of citations from JSON file.
        If path is None, return some default citations.
        
        Args:
            path: Path to JSON file containing corpus of citations
            
        Returns:
            List of citation strings
        """
        if path is None:
            # Return some default citations
            return [
                "[Smith et al., 2020]",
                "[Johnson and Williams, 2019]",
                "[1]",
                "[2]",
                "[3, 4, 5]",
                "[Chen et al., 2018]",
                "[Patel, 2021]",
                "[Wang et al., 2017]",
                "[6]",
                "[7, 8]",
                "[Kumar and Lee, 2022]",
                "[Brown et al., 2016]",
                "[9]",
                "[10, 11]",
                "[Garcia and Rodriguez, 2023]"
            ]
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"Citations corpus file {path} not found, using default citations")
            return self._load_citations_corpus(None)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats of the corpus file
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "citations" in data:
                return data["citations"]
            elif isinstance(data, dict) and "content" in data:
                return data["content"]
            else:
                logger.warning(f"Unrecognized format in citations corpus file {path}, using default citations")
                return self._load_citations_corpus(None)
        except Exception as e:
            logger.error(f"Error loading citations corpus: {str(e)}")
            return self._load_citations_corpus(None)
    
    def predict(self, text: str, num_citations: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict citations for the input text.
        
        Args:
            text: Input text without citations
            num_citations: Number of citations to predict (if None, automatically determined)
            
        Returns:
            Dictionary containing predicted citations and their positions
        """
        # Find potential citation locations
        potential_locations = self._find_potential_locations(text)
        
        # Determine number of citations to insert
        if num_citations is None:
            # Heuristic: insert approximately 1 citation per 500 words
            words = text.split()
            num_citations = max(1, len(words) // 500)
        
        # Select random locations from potential locations
        num_locations = min(num_citations, len(potential_locations))
        selected_locations = random.sample(potential_locations, num_locations)
        selected_locations.sort()  # Sort by position
        
        # Select random citations for each location
        selected_citations = []
        for _ in range(num_locations):
            citation = random.choice(self.citations_corpus)
            selected_citations.append(citation)
        
        # Create result with citations and positions - just store start positions
        citations_content = []
        citations_positions = []
        
        for (start, _), citation in zip(selected_locations, selected_citations):
            citations_content.append(citation)
            citations_positions.append(start)  # Just store the start position
        
        return {
            "citations": {
                "content": citations_content,
                "positions": citations_positions
            }
        }
    
    def _find_potential_locations(self, text: str) -> List[Tuple[int, int]]:
        """
        Find potential citation locations in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) position tuples
        """
        potential_locations = []
        
        # Find all matches of the patterns
        for pattern in self.citation_location_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                potential_locations.append((end, end))  # Insert citation after the match
        
        # Also consider end of sentences as potential locations
        for match in re.finditer(r'[.!?]\s+', text):
            start, end = match.span()
            potential_locations.append((start, start))  # Insert citation before the punctuation
        
        # If no potential locations found, fall back to random paragraph ends
        if not potential_locations:
            for match in re.finditer(r'\n\s*\n', text):
                start, end = match.span()
                potential_locations.append((start, start))  # Insert citation at paragraph end
        
        # If still no locations, use random positions
        if not potential_locations:
            # Insert approximately 1 citation per 500 characters
            num_random_positions = max(1, len(text) // 500)
            for _ in range(num_random_positions):
                pos = random.randint(0, max(0, len(text) - 1))
                potential_locations.append((pos, pos))
        
        return potential_locations

class KeywordBasedPredictor:
    """
    Keyword-based citation predictor.
    Identifies potential citation locations based on keyword patterns and
    matches citations from a corpus.
    """
    
    def __init__(self, citations_corpus_path: Union[str, Path]):
        """
        Initialize the KeywordBasedPredictor.
        
        Args:
            citations_corpus_path: Path to JSON file containing corpus of citations with keywords
        """
        self.citations_corpus = self._load_citations_corpus(citations_corpus_path)
        
        # Patterns for identifying citation contexts
        self.citation_context_patterns = [
            r'\b(?:according to|as shown by|as demonstrated by|as reported by)\b(.{0,50})',
            r'\b(?:previous work|prior work|related work|previous studies|prior studies|related studies)\b(.{0,50})',
            r'\b(?:recently|previously|earlier studies|earlier work)\b(.{0,50})',
            r'\b(?:has been proposed|has been shown|has been demonstrated|has been suggested)\b(.{0,50})',
            r'\s+(?:shows|demonstrates|suggests|indicates|proposes)\s+that\b(.{0,50})',
        ]
    
    def _load_citations_corpus(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load corpus of citations from JSON file.
        Expected format: List of dict with 'citation' and 'keywords' fields.
        
        Args:
            path: Path to JSON file containing corpus of citations
            
        Returns:
            List of citation dictionaries
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Citations corpus file {path} not found, using default citations")
            return self._generate_default_corpus()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
            
            if not isinstance(corpus, list):
                logger.warning("Citations corpus is not a list, using default corpus")
                return self._generate_default_corpus()
            
            # Validate and process corpus entries
            processed_corpus = []
            for item in corpus:
                if isinstance(item, dict) and 'citation' in item and 'keywords' in item:
                    processed_corpus.append(item)
                else:
                    # Try to extract citation and keywords if not directly provided
                    if isinstance(item, dict) and 'text' in item:
                        processed_corpus.append({
                            'citation': item['text'],
                            'keywords': self._extract_keywords(item['text'])
                        })
                    elif isinstance(item, str):
                        processed_corpus.append({
                            'citation': item,
                            'keywords': self._extract_keywords(item)
                        })
            
            if not processed_corpus:
                logger.warning("No valid entries in citations corpus, using default corpus")
                return self._generate_default_corpus()
            
            return processed_corpus
            
        except Exception as e:
            logger.error(f"Error loading citations corpus: {str(e)}")
            return self._generate_default_corpus()
    
    def _generate_default_corpus(self) -> List[Dict[str, Any]]:
        """
        Generate a default citations corpus.
        
        Returns:
            List of citation dictionaries
        """
        return [
            {
                'citation': "[Smith et al., 2020]",
                'keywords': ['neural networks', 'deep learning', 'classification']
            },
            {
                'citation': "[Johnson and Williams, 2019]",
                'keywords': ['natural language processing', 'sentiment analysis']
            },
            {
                'citation': "[Chen et al., 2018]",
                'keywords': ['computer vision', 'object detection', 'image recognition']
            },
            {
                'citation': "[Patel, 2021]",
                'keywords': ['reinforcement learning', 'robotics']
            },
            {
                'citation': "[Wang et al., 2017]",
                'keywords': ['transformer models', 'attention mechanism']
            },
            {
                'citation': "[Kumar and Lee, 2022]",
                'keywords': ['federated learning', 'privacy', 'distributed systems']
            },
            {
                'citation': "[Brown et al., 2016]",
                'keywords': ['graph neural networks', 'knowledge graphs']
            },
            {
                'citation': "[Garcia and Rodriguez, 2023]",
                'keywords': ['large language models', 'few-shot learning']
            }
        ]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract potential keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # This is a very simplistic keyword extraction
        # In a real implementation, use NLP techniques like TF-IDF or keyword extraction models
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        return list(set(word.lower() for word in words))
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict citations for the input text.
        
        Args:
            text: Input text without citations
            
        Returns:
            Dictionary containing predicted citations and their positions
        """
        # Find potential citation contexts
        contexts = self._find_citation_contexts(text)
        
        # Match citations to contexts
        matched_citations = []
        citation_positions = []
        
        for context_start, context_end, context_text in contexts:
            # Find best matching citation
            best_citation = self._match_citation_to_context(context_text)
            if best_citation:
                matched_citations.append(best_citation)
                # Place citation at the end of the context
                pos = context_end
                citation_positions.append(pos)  # Just store the position
        
        return {
            "citations": {
                "content": matched_citations,
                "positions": citation_positions
            }
        }
    
    def _find_citation_contexts(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find potential citation contexts in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end, context_text) tuples
        """
        contexts = []
        
        # Find all matches of the context patterns
        for pattern in self.citation_context_patterns:
            for match in re.finditer(pattern, text):
                full_match = match.group(0)
                context_text = match.group(1) if len(match.groups()) > 0 else full_match
                start, end = match.span()
                contexts.append((start, end, full_match + context_text))
        
        return contexts
    
    def _match_citation_to_context(self, context: str) -> Optional[str]:
        """
        Find the best matching citation for a context.
        
        Args:
            context: Context text
            
        Returns:
            Best matching citation or None
        """
        # Extract keywords from context
        context_keywords = self._extract_keywords(context)
        
        # Score each citation based on keyword overlap
        best_score = -1
        best_citation = None
        
        for citation_entry in self.citations_corpus:
            citation = citation_entry['citation']
            keywords = citation_entry['keywords']
            
            # Count matching keywords
            matching_keywords = set(context_keywords).intersection(keywords)
            score = len(matching_keywords)
            
            if score > best_score:
                best_score = score
                best_citation = citation
        
        # Return best citation if score is above threshold
        if best_score > 0:
            return best_citation
        
        # If no good match, return a random citation
        if self.citations_corpus:
            return random.choice(self.citations_corpus)['citation']
        
        return None

def predict_random_citations(text: str) -> Dict[str, Any]:
    """
    Simple function wrapper for RandomCitationPredictor.
    
    Args:
        text: Input text without citations
        
    Returns:
        Dictionary containing predicted citations and their positions
    """
    predictor = RandomCitationPredictor()
    return predictor.predict(text)
