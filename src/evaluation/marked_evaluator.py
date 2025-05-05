"""
Module for evaluating citation prediction methods using marked citation text.
"""

import os
import re
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkedCitationEvaluator:
    """
    Class for evaluating citation prediction methods using marked citation text.
    This evaluator focuses only on the accuracy of citation content at marked positions.
    """
    
    def __init__(self, test_data_path: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the MarkedCitationEvaluator.
        
        Args:
            test_data_path: Path to test data directory
            output_dir: Directory to save evaluation results
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory
        self.results_dir = self.output_dir / "marked_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.output_dir / "marked_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load test data
        self.test_papers = self._load_test_papers()
    
    def _load_test_papers(self) -> List[Dict[str, Any]]:
        """
        Load test papers data.
        
        Returns:
            List of dictionaries containing test papers data
        """
        # Try different possible locations
        possible_paths = [
            # Try standard location
            self.test_data_path / "test.json",
            self.test_data_path / "test.jsonl",
            
            # Try in splits directory
            self.test_data_path / "splits" / "test.json",
            self.test_data_path / "splits" / "test.jsonl",
            
            # Try in huggingface directory
            self.test_data_path / "huggingface" / "test.jsonl",
        ]
        
        # Try each path
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found test data at {path}")
                
                # JSON file
                if path.suffix == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                
                # JSONL file
                elif path.suffix == '.jsonl':
                    papers = []
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            papers.append(json.loads(line))
                    return papers
        
        # If we get here, no test data was found
        logger.error(f"No test data found in any of the expected locations")
        raise FileNotFoundError(f"Test data not found at {self.test_data_path}")
    
    def evaluate_method(self, 
                       method_name: str, 
                       method_func: Callable[[str], List[str]],
                       parallel: bool = True,
                       n_workers: int = 4) -> Dict[str, Any]:
        """
        Evaluate a citation prediction method on the test set.
        This method works with marked citation text, where each [CITATION] tag needs to be replaced.
        
        Args:
            method_name: Name of the method
            method_func: Function that takes marked text and returns list of citations
            parallel: Whether to process papers in parallel
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating marked citation method: {method_name}")
        
        # Create method results directory
        method_dir = self.results_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Process each test paper
        predictions = []
        
        # Process sequentially
        for paper in tqdm(self.test_papers, desc=f"Evaluating {method_name}"):
            try:
                paper_id = paper["paper_id"]
                
                # Get the marked citation text (with [CITATION] markers)
                marked_text = paper.get("marked_citations_text") or paper.get("expected_text")
                if not marked_text:
                    logger.warning(f"No marked citation text found for paper {paper_id}")
                    continue
                
                # Get ground truth citations
                gt_citations = self._extract_citation_content(paper)
                
                # Get predictions - this should be a list of citations
                # in the same order as they appear in the marked text
                pred_citations = method_func(marked_text)
                
                # Create prediction result
                pred = {
                    "paper_id": paper_id,
                    "pred_citations": pred_citations,
                    "gt_citations": gt_citations
                }
                
                # Save individual prediction
                with open(method_dir / f"{paper_id}_pred.json", 'w', encoding='utf-8') as f:
                    json.dump(pred, f, indent=2)
                
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('paper_id', 'unknown')}: {str(e)}")
    
        # Remove None values (failed predictions)
        predictions = [p for p in predictions if p is not None]
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions)
        
        # Save metrics
        with open(method_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save all predictions
        with open(method_dir / "all_predictions.json", 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
        
        return metrics
    
    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate citation accuracy metrics.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing metrics
        """
        # Aggregated metrics
        total_citations = 0
        total_correct = 0
        total_matching_length = 0
        
        for pred in predictions:
            gt_citations = pred.get("gt_citations", [])
            pred_citations = pred.get("pred_citations", [])
            
            # Count the minimum length to avoid index errors
            min_length = min(len(gt_citations), len(pred_citations))
            total_matching_length += min_length
            
            # If predicted fewer or more citations than ground truth, count as error
            if len(pred_citations) != len(gt_citations):
                logger.warning(
                    f"Number of citations mismatch for paper {pred['paper_id']}: "
                    f"predicted {len(pred_citations)}, ground truth {len(gt_citations)}"
                )
            
            # Count total citations
            total_citations += len(gt_citations)
            
            # Count correct predictions
            for i in range(min_length):
                if self._citations_match(gt_citations[i], pred_citations[i]):
                    total_correct += 1
        
        # Calculate metrics
        citation_accuracy = total_correct / max(1, total_citations)
        length_match_rate = total_matching_length / max(1, total_citations)
        
        return {
            "citation_accuracy": citation_accuracy,
            "length_match_rate": length_match_rate,
            "total_citations": total_citations,
            "total_correct": total_correct
        }
    
    def _citations_match(self, citation1: str, citation2: str) -> bool:
        """
        Check if two citations match.
        This is a simple string comparison.
        
        Args:
            citation1: First citation
            citation2: Second citation
            
        Returns:
            True if citations match, False otherwise
        """
        # Simple string comparison for now
        # In a more sophisticated implementation, you could normalize the citations
        # or use a more advanced matching algorithm
        return citation1.strip() == citation2.strip()
    
    def _extract_citation_content(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract citation content from data.
        
        Args:
            data: Dictionary containing citation data
            
        Returns:
            List of citation content strings
        """
        if "citations" in data and "content" in data["citations"]:
            return data["citations"]["content"]
        
        # Handle different formats
        if "citation_info" in data and "content" in data["citation_info"]:
            return data["citation_info"]["content"]
        
        # Default: empty list if no citations found
        return []
    
    def compare_methods(self, method_names: List[str]) -> None:
        """
        Compare multiple citation prediction methods.
        
        Args:
            method_names: List of method names to compare
        """
        logger.info(f"Comparing methods: {', '.join(method_names)}")
        
        # Load metrics for each method
        methods_metrics = {}
        for method_name in method_names:
            method_metrics_path = self.results_dir / method_name / "metrics.json"
            if not method_metrics_path.exists():
                logger.warning(f"Metrics file not found for method: {method_name}")
                continue
            
            with open(method_metrics_path, 'r', encoding='utf-8') as f:
                methods_metrics[method_name] = json.load(f)
        
        # Create comparison DataFrame
        comparison_data = []
        for method_name, metrics in methods_metrics.items():
            comparison_data.append({
                "Method": method_name,
                "Citation Accuracy": metrics["citation_accuracy"],
                "Length Match Rate": metrics["length_match_rate"],
                "Total Citations": metrics["total_citations"],
                "Total Correct": metrics["total_correct"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "marked_method_comparison.csv", index=False)
        
        # Generate comparison plots
        self._generate_comparison_plots(comparison_df)
    
    def _generate_comparison_plots(self, comparison_df: pd.DataFrame) -> None:
        """
        Generate comparison plots.
        
        Args:
            comparison_df: DataFrame containing comparison data
        """
        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        # Accuracy Comparison
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        sns.barplot(x="Method", y="Citation Accuracy", data=comparison_df)
        
        plt.title("Citation Accuracy Comparison")
        plt.xlabel("Method")
        plt.ylabel("Citation Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "accuracy_comparison.png", dpi=300)
        plt.close()
