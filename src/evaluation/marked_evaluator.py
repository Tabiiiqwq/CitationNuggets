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
import openai

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
        total_citation_positions = 0  # Number of citation positions (slot), in gt
        total_gt_papers = 0  # Total number of cited papers in ground truth, can repeat
        total_pred_papers = 0  # Total number of cited papers in predictions, can repeat
        total_correct_papers = 0  # Total number of correctly cited papers
        total_positions_with_matches = 0  # Number of positions with at least one correct citation
        
        # Set of all unique papers cited
        all_gt_papers = set()
        all_pred_papers = set()
        all_correct_papers = set()
        
        for pred in predictions:
            gt_citations = pred.get("gt_citations", [])
            pred_citations = pred.get("pred_citations", [])
            
            # Ensure gt_citations is a list of lists
            if gt_citations and not isinstance(gt_citations[0], list):
                gt_citations = [[citation] for citation in gt_citations]
            
            # Ensure pred_citations is a list of lists
            if pred_citations and not isinstance(pred_citations[0], list):
                pred_citations = [[citation] for citation in pred_citations]
                
            for i in range(len(gt_citations)):
                position_gt_citations = gt_citations[i]
                total_gt_papers += len(position_gt_citations)
                for citation in position_gt_citations:
                    all_gt_papers.add(citation)
            
            # Count the minimum length to avoid index errors
            min_length = min(len(gt_citations), len(pred_citations))
            
            # If predicted fewer or more citations than ground truth, count as error
            if len(pred_citations) != len(gt_citations):
                logger.warning(
                    f"Number of citation positions mismatch for paper {pred['paper_id']}: "
                    f"predicted {len(pred_citations)}, ground truth {len(gt_citations)}"
                )
            
            # Count total citation positions
            total_citation_positions += len(gt_citations)
            
            # Process each citation position
            for i in range(min_length):
                position_gt_citations = gt_citations[i]
                position_pred_citations = pred_citations[i]
                
                # Count papers at this position
                total_pred_papers += len(position_pred_citations)
                
                # Add to sets of all papers
                for citation in position_pred_citations:
                    all_pred_papers.add(citation)
                
                # Check if any prediction matches any ground truth for this position
                position_has_match = False
                position_correct_count = 0
                
                for pred_citation in position_pred_citations:
                    for gt_citation in position_gt_citations:
                        if self._citations_match(gt_citation, pred_citation):
                            position_correct_count += 1
                            all_correct_papers.add(gt_citation)
                            position_has_match = True
                            break
                
                # Count correctly cited papers at this position
                total_correct_papers += position_correct_count
                
                # Count positions with at least one correct citation
                if position_has_match:
                    total_positions_with_matches += 1
        
        # Calculate metrics
        position_accuracy = total_positions_with_matches / max(1, total_citation_positions)
        paper_precision = total_correct_papers / max(1, total_pred_papers)
        paper_recall = total_correct_papers / max(1, total_gt_papers)
        paper_f1 = 2 * paper_precision * paper_recall / max(1e-6, (paper_precision + paper_recall))
        
        # Calculate overall corpus-level metrics
        corpus_precision = len(all_correct_papers) / max(1, len(all_pred_papers))
        corpus_recall = len(all_correct_papers) / max(1, len(all_gt_papers))
        corpus_f1 = 2 * corpus_precision * corpus_recall / max(1e-6, (corpus_precision + corpus_recall))
        
        return {
            "position_accuracy": position_accuracy,  # Percentage of positions with at least one correct citation
            "paper_precision": paper_precision,  # Precision at the individual citation level. In all predicted papers, how many are correct?
            "paper_recall": paper_recall,  # Recall at the individual citation level
            "paper_f1": paper_f1,  # F1 at the individual citation level
            "corpus_precision": corpus_precision,  # Precision at the corpus level
            "corpus_recall": corpus_recall,  # Recall at the corpus level
            "corpus_f1": corpus_f1,  # F1 at the corpus level
            "total_citation_positions": total_citation_positions,
            "total_gt_papers": total_gt_papers,
            "total_pred_papers": total_pred_papers,
            "total_correct_papers": total_correct_papers,
            "total_positions_with_matches": total_positions_with_matches,
            "unique_gt_papers": len(all_gt_papers),
            "unique_pred_papers": len(all_pred_papers),
            "unique_correct_papers": len(all_correct_papers)
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
        if citation1.strip() == citation2.strip():
            return True
            
        try:

            
            # Initialize OpenAI client
            client = openai.OpenAI()
            
            # Prompt for GPT to compare the citations
            prompt = f"""
            Determine if these two citations refer to the same paper. Focus on the paper title.
            Return only "YES" if they likely refer to the same paper or "NO" if they're different papers.
            
            Citation 1: {citation1}
            Citation 2: {citation2}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise academic citation comparison tool."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
        except Exception as e:
            logger.warning(f"Error in GPT citation comparison: {str(e)}")
            # Fall back to simple string comparison if API call fails
            return citation1.strip() == citation2.strip()
    
    def _extract_citation_content(self, data: Dict[str, Any]) -> List[List[str]]:
        """
        Extract citation content from data.
        
        Args:
            data: Dictionary containing citation data
            
        Returns:
            List of lists of citation content strings, where each inner list
            contains citations for a specific citation position
        """
        # Try to get citations in the expected list of lists format
        if "citations" in data and "content" in data["citations"]:
            content = data["citations"]["content"]
            # Check if content is already in the expected format (list of lists)
            if content and isinstance(content[0], list):
                return content
            # Convert flat list to list of single-item lists
            return [[citation] for citation in content]
        
        # Handle different formats
        if "citation_info" in data and "content" in data["citation_info"]:
            content = data["citation_info"]["content"]
            # Check if content is already in the expected format (list of lists)
            if content and isinstance(content[0], list):
                return content
            # Convert flat list to list of single-item lists
            return [[citation] for citation in content]
        
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
                "Position Accuracy": metrics.get("position_accuracy", 0),
                "Paper F1": metrics.get("paper_f1", 0),
                "Corpus F1": metrics.get("corpus_f1", 0),
                "Paper Precision": metrics.get("paper_precision", 0),
                "Paper Recall": metrics.get("paper_recall", 0),
                "Corpus Precision": metrics.get("corpus_precision", 0),
                "Corpus Recall": metrics.get("corpus_recall", 0),
                "Total Positions": metrics.get("total_citation_positions", 0),
                "Total GT Papers": metrics.get("total_gt_papers", 0),
                "Total Pred Papers": metrics.get("total_pred_papers", 0),
                "Total Correct Papers": metrics.get("total_correct_papers", 0)
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
        
        # Accuracy Metrics Comparison
        plt.figure(figsize=(12, 8))
        
        # Select metrics to plot
        metrics = ["Position Accuracy", "Paper F1", "Corpus F1"]
        
        # Reshape the dataframe for plotting
        plot_data = []
        for _, row in comparison_df.iterrows():
            for metric in metrics:
                plot_data.append({
                    "Method": row["Method"],
                    "Value": row[metric],
                    "Metric": metric
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        sns.barplot(x="Method", y="Value", hue="Metric", data=plot_df)
        
        plt.title("Citation Accuracy Metrics Comparison")
        plt.xlabel("Method")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "accuracy_metrics_comparison.png", dpi=300)
        plt.close()
        
        # Precision-Recall Comparison (Paper Level)
        plt.figure(figsize=(10, 6))
        
        for _, row in comparison_df.iterrows():
            plt.scatter(
                row["Paper Recall"], 
                row["Paper Precision"], 
                s=100, 
                label=row["Method"]
            )
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Paper Recall")
        plt.ylabel("Paper Precision")
        plt.title("Paper-Level Precision-Recall")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "paper_pr_comparison.png", dpi=300)
        plt.close()
        
        # Precision-Recall Comparison (Corpus Level)
        plt.figure(figsize=(10, 6))
        
        for _, row in comparison_df.iterrows():
            plt.scatter(
                row["Corpus Recall"], 
                row["Corpus Precision"], 
                s=100, 
                label=row["Method"]
            )
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Corpus Recall")
        plt.ylabel("Corpus Precision")
        plt.title("Corpus-Level Precision-Recall")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / "corpus_pr_comparison.png", dpi=300)
        plt.close()
