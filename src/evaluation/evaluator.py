"""
Module for evaluating citation prediction methods.
"""

import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .metrics import CitationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CitationMethodEvaluator:
    """
    Class for evaluating and comparing different citation prediction methods.
    """
    
    def __init__(self, test_data_path: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the CitationMethodEvaluator.
        
        Args:
            test_data_path: Path to test data directory
            output_dir: Directory to save evaluation results
        """
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
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
                       method_func: Callable[[str], Dict[str, Any]],
                       parallel: bool = True,
                       n_workers: int = 4) -> Dict[str, Any]:
        """
        Evaluate a citation prediction method on the test set.
        
        Args:
            method_name: Name of the method
            method_func: Function that takes input text and returns citations
            parallel: Whether to process papers in parallel
            n_workers: Number of parallel workers
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating method: {method_name}")
        
        # Create method results directory
        method_dir = self.results_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Process each test paper
        predictions = []
        
        if parallel:
            # Define worker function
            def process_paper(paper):
                try:
                    paper_id = paper["paper_id"]
                    input_text = paper["no_citations_text"] if "no_citations_text" in paper else paper["input_text"]
                    
                    # Get predictions
                    pred = method_func(input_text)
                    
                    # Add paper_id to predictions
                    pred["paper_id"] = paper_id
                    
                    # Save individual prediction
                    with open(method_dir / f"{paper_id}_pred.json", 'w', encoding='utf-8') as f:
                        json.dump(pred, f, indent=2)
                    
                    return pred
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('paper_id', 'unknown')}: {str(e)}")
                    return None
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                predictions = list(tqdm(
                    executor.map(process_paper, self.test_papers),
                    total=len(self.test_papers),
                    desc=f"Evaluating {method_name}"
                ))
        else:
            # Process sequentially
            for paper in tqdm(self.test_papers, desc=f"Evaluating {method_name}"):
                try:
                    paper_id = paper["paper_id"]
                    input_text = paper["no_citations_text"] if "no_citations_text" in paper else paper["input_text"]
                    
                    # Get predictions
                    pred = method_func(input_text)
                    
                    # Add paper_id to predictions
                    pred["paper_id"] = paper_id
                    
                    # Save individual prediction
                    with open(method_dir / f"{paper_id}_pred.json", 'w', encoding='utf-8') as f:
                        json.dump(pred, f, indent=2)
                    
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Error processing paper {paper.get('paper_id', 'unknown')}: {str(e)}")
        
        # Remove None values (failed predictions)
        predictions = [p for p in predictions if p is not None]
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(predictions, self.test_papers)
        
        # Save metrics
        with open(method_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save all predictions
        with open(method_dir / "all_predictions.json", 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
        
        return metrics
    
    def _calculate_aggregate_metrics(self, 
                                   predictions: List[Dict[str, Any]], 
                                   ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all test papers.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            
        Returns:
            Dictionary containing aggregate metrics
        """
        # Create lookup for ground truth by paper_id
        gt_by_id = {paper["paper_id"]: paper for paper in ground_truth}
        
        # Aggregate citation metrics
        total_gt_citations = 0
        total_pred_citations = 0
        total_correct_citations = 0
        
        # Aggregate position metrics
        total_gt_positions = 0
        total_pred_positions = 0
        total_correct_positions = 0
        total_position_error = 0
        matched_positions = 0
        
        # Position threshold
        position_threshold = 50  # Characters
        
        for pred in predictions:
            paper_id = pred["paper_id"]
            if paper_id not in gt_by_id:
                logger.warning(f"Ground truth not found for paper_id: {paper_id}")
                continue
            
            gt = gt_by_id[paper_id]
            
            # Extract citations
            gt_citations = set(self._extract_citation_content(gt))
            pred_citations = set(self._extract_citation_content(pred))
            
            # Count citations
            total_gt_citations += len(gt_citations)
            total_pred_citations += len(pred_citations)
            total_correct_citations += len(gt_citations.intersection(pred_citations)) # TODO: change to judge if is the same paper title
            
            # Extract positions
            gt_positions = self._extract_citation_positions(gt)
            pred_positions = self._extract_citation_positions(pred)
            
            # Count positions
            total_gt_positions += len(gt_positions)
            total_pred_positions += len(pred_positions)
            
            # Count correct positions and calculate error
            matched_gt_positions = set()
            
            for pred_pos in pred_positions:
                min_error = float('inf')
                closest_gt_idx = None
                
                for i, gt_pos in enumerate(gt_positions):
                    # Check if positions are close
                    if abs(pred_pos - gt_pos) <= position_threshold:
                        # Calculate error
                        error = abs(pred_pos - gt_pos)
                        
                        if error < min_error:
                            min_error = error
                            closest_gt_idx = i
                
                if closest_gt_idx is not None and closest_gt_idx not in matched_gt_positions:
                    total_correct_positions += 1
                    matched_gt_positions.add(closest_gt_idx)
                    total_position_error += min_error
                    matched_positions += 1
        
        # Calculate metrics
        citation_precision = total_correct_citations / max(1, total_pred_citations)
        citation_recall = total_correct_citations / max(1, total_gt_citations)
        citation_f1 = 2 * citation_precision * citation_recall / max(1e-6, (citation_precision + citation_recall))
        
        position_precision = total_correct_positions / max(1, total_pred_positions)
        position_recall = total_correct_positions / max(1, total_gt_positions)
        position_f1 = 2 * position_precision * position_recall / max(1e-6, (position_precision + position_recall))
        
        avg_position_error = total_position_error / max(1, matched_positions)
        
        return {
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_f1": citation_f1,
            "position_precision": position_precision,
            "position_recall": position_recall,
            "position_f1": position_f1,
            "avg_position_error": avg_position_error,
            "total_gt_citations": total_gt_citations,
            "total_pred_citations": total_pred_citations,
            "total_correct_citations": total_correct_citations,
            "total_gt_positions": total_gt_positions,
            "total_pred_positions": total_pred_positions,
            "total_correct_positions": total_correct_positions,
        }
    
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
    
    def _extract_citation_positions(self, data: Dict[str, Any]) -> List[int]:
        """
        Extract citation positions from data.
        
        Args:
            data: Dictionary containing citation data
            
        Returns:
            List of position integers
        """
        if "citations" in data and "positions" in data["citations"]:
            return data["citations"]["positions"]
        
        # Handle different formats
        if "citation_info" in data and "positions" in data["citation_info"]:
            return data["citation_info"]["positions"]
        
        # Default: empty list if no positions found
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
                "Citation Precision": metrics["citation_precision"],
                "Citation Recall": metrics["citation_recall"],
                "Citation F1": metrics["citation_f1"],
                "Position Precision": metrics["position_precision"],
                "Position Recall": metrics["position_recall"],
                "Position F1": metrics["position_f1"],
                "Avg Position Error": metrics["avg_position_error"],
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "method_comparison.csv", index=False)
        
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
        
        # F1 Score Comparison
        plt.figure(figsize=(10, 6))
        metrics = ["Citation F1", "Position F1"]
        
        # Reshape the dataframe for plotting
        plot_data = []
        for _, row in comparison_df.iterrows():
            for metric in metrics:
                plot_data.append({
                    "Method": row["Method"],
                    "F1 Score": row[metric],
                    "Type": metric.split()[0]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        ax = sns.barplot(x="Method", y="F1 Score", hue="Type", data=plot_df)
        
        plt.title("F1 Score Comparison")
        plt.xlabel("Method")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "f1_comparison.png", dpi=300)
        plt.close()
        
        # Precision-Recall Comparison
        plt.figure(figsize=(12, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Citation Precision-Recall
        for _, row in comparison_df.iterrows():
            ax1.scatter(row["Citation Recall"], row["Citation Precision"], s=100, label=row["Method"])
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Citation Recall")
        ax1.set_ylabel("Citation Precision")
        ax1.set_title("Citation Precision-Recall")
        ax1.grid(True)
        ax1.legend()
        
        # Position Precision-Recall
        for _, row in comparison_df.iterrows():
            ax2.scatter(row["Position Recall"], row["Position Precision"], s=100, label=row["Method"])
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Position Recall")
        ax2.set_ylabel("Position Precision")
        ax2.set_title("Position Precision-Recall")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "precision_recall_comparison.png", dpi=300)
        plt.close()
        
        # Position Error Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Method", y="Avg Position Error", data=comparison_df)
        plt.title("Average Position Error Comparison")
        plt.xlabel("Method")
        plt.ylabel("Average Position Error (characters)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "position_error_comparison.png", dpi=300)
        plt.close()
