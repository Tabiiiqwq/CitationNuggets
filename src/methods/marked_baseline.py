"""
Module providing baseline citation prediction methods for marked citation text.
"""

import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RandomMarkedCitationPredictor:
    """
    Random baseline predictor for marked citations.
    This predictor picks random citations from a corpus for each [CITATION] marker.
    """
    
    def __init__(self, citations_corpus_path: Optional[Union[str, Path]] = None):
        """
        Initialize the RandomMarkedCitationPredictor.
        
        Args:
            citations_corpus_path: Path to JSON file containing corpus of citations to sample from
        """
        self.citations_corpus = self._load_citations_corpus(citations_corpus_path)
    
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
    
    def predict(self, marked_text: str) -> List[List[str]]:
        """
        Predict citations for the input text with [CITATION] markers.
        
        Args:
            marked_text: Text with [CITATION] markers
            
        Returns:
            List of lists of predicted citations in the same order as the markers.
            Each inner list contains one or more citations for a specific marker.
        """
        # Count number of [CITATION] markers
        citation_markers = re.findall(r'\[CITATION\]', marked_text)
        num_citations = len(citation_markers)
        
        # Select random citations
        predictions = []
        for _ in range(num_citations):
            # Randomly decide how many citations to include (1-3)
            num_citations_at_position = random.randint(1, min(3, len(self.citations_corpus)))
            
            # Select random citations for this position (without duplicates)
            position_citations = random.sample(
                self.citations_corpus, 
                num_citations_at_position
            )
            
            predictions.append(position_citations)
        
        return predictions


class ContextBasedMarkedCitationPredictor:
    """
    Context-based predictor for marked citations.
    This predictor uses the surrounding context to pick citations from a corpus.
    """
    
    def __init__(self, citations_corpus_path: Union[str, Path]):
        """
        Initialize the ContextBasedMarkedCitationPredictor.
        
        Args:
            citations_corpus_path: Path to JSON file containing corpus of citations with keywords
        """
        self.citations_corpus = self._load_citations_corpus(citations_corpus_path)
    
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
            logger.warning(f"Citations corpus file {path} not found, using default corpus")
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
        # Very simplistic keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        return list(set(word.lower() for word in words))
    
    def predict(self, marked_text: str) -> List[List[str]]:
        """
        Predict citations for the input text with [CITATION] markers.
        
        Args:
            marked_text: Text with [CITATION] markers
            
        Returns:
            List of lists of predicted citations in the same order as the markers
            Each inner list contains one or more citations for a specific marker
        """
        # Find all citation markers with their surrounding context
        citation_contexts = []
        markers_pattern = r'(.{0,100})\[CITATION\](.{0,100})'
        
        for match in re.finditer(markers_pattern, marked_text):
            before_text = match.group(1)
            after_text = match.group(2)
            context = before_text + after_text
            citation_contexts.append(context)
        
        # Match citations to contexts
        predictions = []
        for context in citation_contexts:
            # Find the top matching citations for this context
            top_citations = self._match_citations_to_context(context)
            predictions.append(top_citations)
        
        return predictions
    
    def _match_citations_to_context(self, context: str) -> List[str]:
        """
        Find multiple matching citations for a context.
        
        Args:
            context: Context text
            
        Returns:
            List of best matching citations
        """
        # Extract keywords from context
        context_keywords = self._extract_keywords(context)
        
        # Score each citation based on keyword overlap
        scored_citations = []
        
        for citation_entry in self.citations_corpus:
            citation = citation_entry['citation']
            keywords = citation_entry['keywords']
            
            # Count matching keywords
            matching_keywords = set(context_keywords).intersection(keywords)
            score = len(matching_keywords)
            
            scored_citations.append((citation, score))
        
        # Sort by score in descending order
        scored_citations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 1-3 citations based on score
        num_citations = min(3, len(scored_citations))
        
        # Only return citations with a positive score, or at least one random citation if none match
        matching_citations = [cit for cit, score in scored_citations[:num_citations] if score > 0]
        
        if not matching_citations and self.citations_corpus:
            # If no good matches, return a single random citation
            return [random.choice(self.citations_corpus)['citation']]
        
        return matching_citations

def predict_random_marked_citations(marked_text: str) -> List[List[str]]:
    """
    Simple function wrapper for RandomMarkedCitationPredictor.
    
    Args:
        marked_text: Text with [CITATION] markers
        
    Returns:
        List of lists of predicted citations, one list per citation marker
    """
    predictor = RandomMarkedCitationPredictor()
    return predictor.predict(marked_text)

def predict_context_marked_citations(marked_text: str) -> List[List[str]]:
    """
    Simple function wrapper for ContextBasedMarkedCitationPredictor.
    Generates a default corpus internally.
    
    Args:
        marked_text: Text with [CITATION] markers
        
    Returns:
        List of lists of predicted citations, one list per citation marker
    """
    # Create with default corpus by passing a dummy path that doesn't exist
    dummy_path = Path("nonexistent_path.json")
    predictor = ContextBasedMarkedCitationPredictor(dummy_path)
    return predictor.predict(marked_text)
