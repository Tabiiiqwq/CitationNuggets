"""
Module providing LLM-based citation prediction methods.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable

# Optional imports for LLM integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAICitationPredictor:
    """
    Citation predictor using OpenAI API.
    Uses a large language model to predict citations for academic papers.
    """
    
    def __init__(self, 
                model: str = "gpt-4o",
                api_key: Optional[str] = None,
                references_path: Optional[Union[str, Path]] = None,
                temperature: float = 0.2,
                max_tokens: int = 4000):
        """
        Initialize the OpenAICitationPredictor.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if None, uses environment variable)
            references_path: Path to JSON file containing references corpus
            temperature: Sampling temperature for the model
            max_tokens: Maximum tokens for model response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAICitationPredictor")
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load references corpus if provided
        self.references = self._load_references(references_path) if references_path else None
    
    def _load_references(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load references corpus from JSON file.
        
        Args:
            path: Path to JSON file containing references corpus
            
        Returns:
            List of reference dictionaries
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"References file {path} not found")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading references: {str(e)}")
            return []
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict citations for the input text using OpenAI API.
        
        Args:
            text: Input text without citations
            
        Returns:
            Dictionary containing predicted citations and their positions
        """
        # Truncate text if it's too long
        max_text_length = 16000  # Adjust based on model token limits
        if len(text) > max_text_length:
            logger.warning(f"Text length ({len(text)}) exceeds limit, truncating to {max_text_length} characters")
            text = text[:max_text_length]
        
        # Build system prompt
        system_prompt = """
You are an academic citation assistant that helps insert citations into scholarly papers. 
The input is a paper with all citations removed. Your task is to:

1. Identify statements that should be cited (claims, references to prior work, etc.)
2. Insert appropriate citations at these positions using [CITATION] as a placeholder
3. For each citation, provide its position (character start/end) and content

Respond ONLY with a JSON object containing:
1. "citations" object with "content" (array of citation strings) and "positions" (array of [start, end] integer pairs)
"""

        user_prompt = f"""
Please analyze the following paper and identify where citations should be added.

Paper:
{text}

Identify places where citations should be added and respond with a JSON object.
"""

        # Add references context if available
        if self.references:
            references_text = "\n\n".join([
                f"Reference {i+1}: {ref.get('title', 'Untitled')}, {ref.get('authors', 'Unknown')}, {ref.get('year', '')}"
                for i, ref in enumerate(self.references[:20])  # Limit to 20 references
            ])
            user_prompt += f"\n\nHere are some potential references to use:\n{references_text}"

        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            
            # Parse JSON from response
            citation_data = self._extract_json_from_response(response_text)
            
            # Validate the structure
            if not citation_data or "citations" not in citation_data:
                logger.warning("Failed to get valid citation data from model")
                return {"citations": {"content": [], "positions": []}}
            
            if "content" not in citation_data["citations"] or "positions" not in citation_data["citations"]:
                logger.warning("Citation data missing content or positions")
                return {"citations": {"content": [], "positions": []}}
            
            # Ensure positions are in the right format
            try:
                positions = citation_data["citations"]["positions"]
                for i, pos in enumerate(positions):
                    if isinstance(pos, list) and len(pos) == 2:
                        positions[i] = (int(pos[0]), int(pos[1]))
                    else:
                        positions[i] = (0, 0)
                        logger.warning(f"Invalid position format: {pos}")
                
                citation_data["citations"]["positions"] = positions
            except Exception as e:
                logger.error(f"Error processing positions: {str(e)}")
                citation_data["citations"]["positions"] = []
            
            return citation_data
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"citations": {"content": [], "positions": []}}
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON object from model response.
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Extracted JSON object as dictionary
        """
        try:
            # Try to parse entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from within text
            try:
                # Look for JSON between triple backticks
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Look for JSON between curly braces
                json_match = re.search(r"(\{[\s\S]*\})", response_text)
                if json_match:
                    return json.loads(json_match.group(1))
                
                logger.warning("Could not extract JSON from response")
                return {}
            except Exception as e:
                logger.error(f"Error extracting JSON: {str(e)}")
                return {}

class LangchainCitationPredictor:
    """
    Citation predictor using LangChain with OpenAI.
    Uses a structured approach with LangChain to predict citations.
    """
    
    def __init__(self, 
                model: str = "gpt-4o",
                api_key: Optional[str] = None,
                references_path: Optional[Union[str, Path]] = None,
                temperature: float = 0.2):
        """
        Initialize the LangchainCitationPredictor.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if None, uses environment variable)
            references_path: Path to JSON file containing references corpus
            temperature: Sampling temperature for the model
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain package is required for LangchainCitationPredictor")
        
        # Set API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            verbose=True
        )
        
        # Load references corpus if provided
        self.references = self._load_references(references_path) if references_path else None
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an academic citation assistant that helps insert citations into scholarly papers. 
The input is a paper with all citations removed. Your task is to:

1. Identify statements that should be cited (claims, references to prior work, etc.)
2. Insert appropriate citations at these positions using [CITATION] as a placeholder
3. For each citation, provide its position (character start/end) and content

Respond ONLY with a JSON object containing:
1. "citations" object with "content" (array of citation strings) and "positions" (array of [start, end] integer pairs)
"""),
            ("human", """
Please analyze the following paper and identify where citations should be added.

Paper:
{text}

{references_context}

Identify places where citations should be added and respond with a JSON object.
"""),
        ])
        
        # Create chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def _load_references(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load references corpus from JSON file.
        
        Args:
            path: Path to JSON file containing references corpus
            
        Returns:
            List of reference dictionaries
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"References file {path} not found")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading references: {str(e)}")
            return []
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict citations for the input text using LangChain.
        
        Args:
            text: Input text without citations
            
        Returns:
            Dictionary containing predicted citations and their positions
        """
        # Truncate text if it's too long
        max_text_length = 16000  # Adjust based on model token limits
        if len(text) > max_text_length:
            logger.warning(f"Text length ({len(text)}) exceeds limit, truncating to {max_text_length} characters")
            text = text[:max_text_length]
        
        # Prepare references context
        references_context = ""
        if self.references:
            references_text = "\n\n".join([
                f"Reference {i+1}: {ref.get('title', 'Untitled')}, {ref.get('authors', 'Unknown')}, {ref.get('year', '')}"
                for i, ref in enumerate(self.references[:20])  # Limit to 20 references
            ])
            references_context = f"Here are some potential references to use:\n{references_text}"
        
        try:
            # Run chain
            result = self.chain.run(text=text, references_context=references_context)
            
            # Parse JSON from response
            citation_data = self._extract_json_from_response(result)
            
            # Validate the structure
            if not citation_data or "citations" not in citation_data:
                logger.warning("Failed to get valid citation data from model")
                return {"citations": {"content": [], "positions": []}}
            
            if "content" not in citation_data["citations"] or "positions" not in citation_data["citations"]:
                logger.warning("Citation data missing content or positions")
                return {"citations": {"content": [], "positions": []}}
            
            # Ensure positions are in the right format
            try:
                positions = citation_data["citations"]["positions"]
                for i, pos in enumerate(positions):
                    if isinstance(pos, list) and len(pos) == 2:
                        positions[i] = (int(pos[0]), int(pos[1]))
                    else:
                        positions[i] = (0, 0)
                        logger.warning(f"Invalid position format: {pos}")
                
                citation_data["citations"]["positions"] = positions
            except Exception as e:
                logger.error(f"Error processing positions: {str(e)}")
                citation_data["citations"]["positions"] = []
            
            return citation_data
            
        except Exception as e:
            logger.error(f"Error running LangChain: {str(e)}")
            return {"citations": {"content": [], "positions": []}}
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract JSON object from model response.
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Extracted JSON object as dictionary
        """
        import re
        
        try:
            # Try to parse entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from within text
            try:
                # Look for JSON between triple backticks
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Look for JSON between curly braces
                json_match = re.search(r"(\{[\s\S]*\})", response_text)
                if json_match:
                    return json.loads(json_match.group(1))
                
                logger.warning("Could not extract JSON from response")
                return {}
            except Exception as e:
                logger.error(f"Error extracting JSON: {str(e)}")
                return {}

def predict_with_openai(text: str) -> Dict[str, Any]:
    """
    Simple function wrapper for OpenAICitationPredictor.
    
    Args:
        text: Input text without citations
        
    Returns:
        Dictionary containing predicted citations and their positions
    """
    if not OPENAI_AVAILABLE:
        logger.error("openai package not available")
        return {"citations": {"content": [], "positions": []}}
    
    try:
        predictor = OpenAICitationPredictor()
        return predictor.predict(text)
    except Exception as e:
        logger.error(f"Error using OpenAI predictor: {str(e)}")
        return {"citations": {"content": [], "positions": []}}
