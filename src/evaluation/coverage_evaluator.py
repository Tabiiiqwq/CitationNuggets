import os
import re
import json
import logging
from enum import Enum
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

logging.getLogger("httpx").setLevel(logging.WARNING)

class TestTaskType(Enum):
    PAPER = "PAPER"
    PARAGRAPH = "PARAGRAPH"
    SECTION = "SECTION"

class CoverageCitationEvaluator:
    
    def __init__(self, test_data_path: Union[str, Path], output_dir: Union[str, Path], test_tasks: List[TestTaskType]):
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_tasks = test_tasks
        
        # Create results directory
        self.results_dir = self.output_dir / "coverage_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test data
        self.test_papers = self._load_test_papers()

        logger.info(f"Loaded {len(self.test_papers)} test papers from {self.test_data_path}")
        logger.info(f"Test tasks: {', '.join([task.value for task in self.test_tasks])}")
    
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
    

    def _calculate_coverage(self, predictions: List[str], gt: List[dict]) -> Tuple[float, int]:

        covered = 0
        for citation in gt:
            for citation_pred in predictions:
                if self._citations_match(citation_pred, str(citation)):
                    covered += 1
                    break

        # print(f"Covered {covered} out of {len(gt)} citations")
        return covered / len(gt), covered

    def _calculate_coverage_faster(self, predictions: List[str], gt: List[dict]) -> Tuple[float, int]:
        """
        Calculate citation coverage using LLM to batch process comparisons.
        This is more efficient than comparing each ground truth citation with each prediction.
        
        Args:
            predictions: List of predicted citations
            gt: List of ground truth citations
            
        Returns:
            Tuple of (coverage_rate, number_of_covered_citations)
        """
        client = openai.OpenAI()
        
        # Create system prompt with all predictions
        predictions_text = "\n".join([f"{i+1}. {pred}" for i, pred in enumerate(predictions)])
        system_prompt = f"""You are a citation matching expert. You will be provided with a list of predicted citations followed by a ground truth citation.
        Your task is to determine if the ground truth citation matches any of the predicted citations in the list.
        The citations may have different formats but should refer to the same work (same title, authors etc.).
        Respond with "YES" if there is a match or "NO" if there is no match.

        Here are the predicted citations:
        {predictions_text}
        """
        
        covered = 0
        for citation in gt:
            try:
                # User prompt with the ground truth citation
                user_prompt = f"Ground truth citation:\n {str(citation)}\n\nDoes this citation match any of the predicted citations listed above?"
                
                # Call the LLM to check for a match
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                )
                
                result = response.choices[0].message.content.strip().upper()
                if "YES" in result:
                    covered += 1
                    # logger.info(f"Matched citation: {str(citation)}")
                    
            except Exception as e:
                logger.warning(f"Error in batch citation comparison: {str(e)}")
                # Fall back to regular comparison method for this citation
                for citation_pred in predictions:
                    if self._citations_match(citation_pred, str(citation)):
                        covered += 1
                        break

        return (covered / len(gt), covered) if len(gt) > 0 else (0, 0)
                

    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate citation accuracy metrics and citation counts.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing metrics and citation counts
        """
        # Coverage values
        paper_coverage = 0
        section_coverage = 0
        paragraph_coverage = 0
        
        # Citation counts
        paper_gt_citations = 0
        paper_pred_citations = 0
        paper_covered_citations = 0
        
        section_gt_citations = 0
        section_pred_citations = 0
        section_covered_citations = 0
        
        paragraph_gt_citations = 0
        paragraph_pred_citations = 0
        paragraph_covered_citations = 0
        
        # Number of items processed
        paper_count = 0
        section_count = 0
        paragraph_count = 0
        
        for paper in predictions:
            if TestTaskType.PAPER in self.test_tasks:
                pred_citation = paper['paper_pred'][0]
                pred_gt = paper['paper_gt'][0]
                
                # Count citations
                paper_gt_citations += len(pred_gt)
                paper_pred_citations += len(pred_citation)
                
                # Calculate coverage
                if len(pred_gt) > 0:
                    # coverage_rate, covered_count = self._calculate_coverage(pred_citation, pred_gt)
                    coverage_rate, covered_count = self._calculate_coverage_faster(pred_citation, pred_gt)
                    
                    paper_coverage += coverage_rate
                    paper_covered_citations += covered_count
                    paper_count += 1

            if TestTaskType.SECTION in self.test_tasks and 'section_preds' in paper and 'section_gts' in paper:
                pred_citations = paper['section_preds']
                pred_gts = paper['section_gts']
                
                if pred_gts and len(pred_gts) > 0:
                    for i in range(len(pred_gts)):
                        # Count citations
                        section_gt_citations += len(pred_gts[i])
                        section_pred_citations += len(pred_citations[i])
                        
                        # Calculate coverage
                        if len(pred_gts[i]) > 0:
                            # coverage_rate, coverd_cout = self._calculate_coverage(pred_citations[i], pred_gts[i])
                            coverage_rate, coverd_cout = self._calculate_coverage_faster(pred_citations[i], pred_gts[i])
                            
                            section_coverage += coverage_rate 
                            section_covered_citations += coverd_cout
                            section_count += 1

            if TestTaskType.PARAGRAPH in self.test_tasks and 'paragraph_preds' in paper and 'paragraph_gts' in paper:
                pred_citations = paper['paragraph_preds']
                pred_gts = paper['paragraph_gts']
                
                if pred_gts and len(pred_gts) > 0:
                    for i in range(len(pred_gts)):
                        # Count citations
                        paragraph_gt_citations += len(pred_gts[i])
                        paragraph_pred_citations += len(pred_citations[i])
                        
                        # Calculate coverage
                        if len(pred_gts[i]) > 0:
                            # coverage_rate, covered_count = self._calculate_coverage(pred_citations[i], pred_gts[i])
                            coverage_rate, covered_count = self._calculate_coverage_faster(pred_citations[i], pred_gts[i])
                            
                            paragraph_coverage += coverage_rate
                            paragraph_covered_citations += covered_count                            
                            paragraph_count += 1
                
        # Calculate average coverage: average per paper/section/paragraph
        avg_paper_coverage = paper_coverage / paper_count if paper_count > 0 else 0
        avg_section_coverage = section_coverage / section_count if section_count > 0 else 0
        avg_paragraph_coverage = paragraph_coverage / paragraph_count if paragraph_count > 0 else 0
        
        return {
            # Coverage metrics
            "paper_coverage": avg_paper_coverage,
            "section_coverage": avg_section_coverage,
            "paragraph_coverage": avg_paragraph_coverage,
            
            # Citation counts
            "paper_gt_citations": paper_gt_citations,
            "paper_pred_citations": paper_pred_citations,
            "paper_covered_count": paper_covered_citations,
            "section_gt_citations": section_gt_citations,
            "section_pred_citations": section_pred_citations, 
            "section_covered_count": section_covered_citations,
            "paragraph_gt_citations": paragraph_gt_citations,
            "paragraph_pred_citations": paragraph_pred_citations,
            "paragraph_covered_count": paragraph_covered_citations,
            
            # Item counts
            "paper_count": paper_count,
            "section_count": section_count,
            "paragraph_count": paragraph_count
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
            Determine if these two citations refer to the same paper.
            They may have different formats, but they should refer to the same work. Like: same title, same shortened title, same tag like("et al.")...
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

    def _visualize_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Create visualizations comparing the performance of different methods.
        
        Args:
            comparison_df: DataFrame containing method comparison data
        """
        # Create output directory for plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
        # Plot coverage metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        coverage_metrics = ["Paper Coverage", "Section Coverage", "Paragraph Coverage"]
        coverage_df = comparison_df.melt(
            id_vars=["Method"], 
            value_vars=coverage_metrics,
            var_name="Metric", 
            value_name="Coverage"
        )
        
        sns.barplot(x="Method", y="Coverage", hue="Metric", data=coverage_df, ax=ax)
        ax.set_title("Coverage Comparison Across Methods")
        ax.set_xlabel("Method")
        ax.set_ylabel("Coverage Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "coverage_comparison.png", dpi=300)
        plt.close()
        
    def evaluate_method(self, 
                       method_name: str, 
                       method_func: Callable[[str], List[str]],
                       n_papers=None) -> Dict[str, Any]:
        """
        Evaluate a citation prediction method on the test set.
        This method works with marked citation text, where each [CITATION] tag needs to be replaced.
        
        Args:
            method_name: Name of the method
            method_func: Function that takes marked text and returns list of citations
            parallel: Whether to process papers in parallel
            
        Returns:
            Dictionary containing evaluation results
        """
        # logger.info(f"Evaluating method: {method_name}")
        
        # Create method results directory
        method_dir = self.results_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Process each test paper
        predictions = []
        
        # Process sequentially
        for idx, paper in tqdm(enumerate(self.test_papers), desc=f"Evaluating {method_name}"):

            if n_papers is not None and idx >= n_papers:
                break
            
            try:
                paper_id = paper["paper_id"]
                # logger.info(f"Processing: {paper_id} ({idx + 1}/{len(self.test_papers)})")
                
                meta = {
                    "title": paper['full_paper']['title'],
                    "section_ids": [], # list of tuples (section_id, section_title)
                    "paragraph_ids": [], # list of str
                }

                paper_pred, paper_gt = [], []  # Placeholder for paper-level predictions and ground truth
                if TestTaskType.PAPER in self.test_tasks:
                    paper_data = paper['full_paper']
                    masked_text = paper_data['masked_text']
                    paper_pred = method_func(masked_text)
                    paper_gt = paper_data['citations']

                section_preds, section_gts = [], []  # Placeholder for section-level predictions and ground truth
                if TestTaskType.SECTION in self.test_tasks:
                    for section_key, section_data in paper['sections'].items():
                        section_gt = section_data['citations']
                        if len(section_gt) == 0: continue

                        masked_text = section_data['masked_text']
                        section_pred = method_func(masked_text)

                        section_preds.append(section_pred)
                        section_gts.append(section_gt)

                        meta['section_ids'].append((section_key, section_data['title']))

                

                paragraph_preds, paragraph_gts = [], []
                if TestTaskType.PARAGRAPH in self.test_tasks:
                    for paragraph_key, paragraph_data in paper['paragraphs'].items():
                        paragraph_gt = paragraph_data['citations']
                        if len(paragraph_gt) == 0: continue

                        masked_text = paragraph_data['masked_text']
                        paragraph_pred = method_func(masked_text)

                        paragraph_preds.append(paragraph_pred)
                        paragraph_gts.append(paragraph_gt)

                        meta['paragraph_ids'].append(paragraph_key)

        
                
                # Create prediction result
                pred = {
                    "paper_id": paper_id,
                    "meta": meta,
                    "paper_pred": [paper_pred], # list of list
                    "paper_gt": [paper_gt], # list of list
                    "section_preds": section_preds, # list of list
                    "section_gts": section_gts, # list of list
                    "paragraph_preds": paragraph_preds,  # list of list
                    "paragraph_gts": paragraph_gts,  # list of list
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
        logger.info(f"Calculating metrics for {method_name}")
        metrics = self._calculate_metrics(predictions)
        
        # Save metrics
        with open(method_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save all predictions
        with open(method_dir / "all_predictions.json", 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
        
        return metrics
    
    def compare_methods(self, method_names: List[str]) -> None:
        """
        Compare multiple citation prediction methods based on coverage metrics.
        
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
                # Coverage metrics
                "Paper Coverage": metrics.get("paper_coverage", 0),
                "Section Coverage": metrics.get("section_coverage", 0),
                "Paragraph Coverage": metrics.get("paragraph_coverage", 0),
                
                # Paper metrics
                "Paper GT Citations": metrics.get("paper_gt_citations", 0),
                "Paper Pred Citations": metrics.get("paper_pred_citations", 0),
                "Paper Covered Count": metrics.get("paper_covered_count", 0),
                
                # Section metrics
                "Section GT Citations": metrics.get("section_gt_citations", 0),
                "Section Pred Citations": metrics.get("section_pred_citations", 0),
                "Section Covered Count": metrics.get("section_covered_count", 0),
                
                # Paragraph metrics
                "Paragraph GT Citations": metrics.get("paragraph_gt_citations", 0),
                "Paragraph Pred Citations": metrics.get("paragraph_pred_citations", 0),
                "Paragraph Covered Count": metrics.get("paragraph_covered_count", 0),
                
                # Count metrics
                "Paper Count": metrics.get("paper_count", 0),
                "Section Count": metrics.get("section_count", 0),
                "Paragraph Count": metrics.get("paragraph_count", 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / "coverage_method_comparison.csv", index=False)
        
        # Create and save visualizations
        if len(comparison_data) > 0:
            self._visualize_comparison(comparison_df)


def eval_with_json_result(json_path: Union[str, Path], 
                          output_dir: Union[str, Path], 
                          test_data_path: Union[str, Path],
                          test_tasks: List[TestTaskType],
                          method_name: str = "json_method"):
    # Convert string task names to TestTaskType enum values
    if test_tasks is None:
        test_tasks = ["PAPER", "SECTION", "PARAGRAPH"]
        
    
    # Initialize evaluator
    evaluator = CoverageCitationEvaluator(
        test_data_path=test_data_path,
        output_dir=output_dir,
        test_tasks=test_tasks
    )
    
    # Create method results directory
    method_dir = Path(output_dir) / "coverage_results" / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions from JSON file
    logger.info(f"Loading predictions from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Calculate metrics
    logger.info(f"Calculating metrics for predictions from {json_path}")
    metrics = evaluator._calculate_metrics(predictions)
    
    # Save metrics
    with open(method_dir / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {method_dir}")
    
    return metrics
    


if __name__ == "__main__":
    evaluator = CoverageCitationEvaluator(
        test_data_path=r"data\output\dataset_coverage",
        output_dir="eval/test_coverage",
        test_tasks=[TestTaskType.SECTION]
    )
    
    # Define a dummy method function for testing
    def dummy_method(marked_text: str) -> List[str]:
        # This is a placeholder for the actual method logic
        return ["Gemini: A family of highly capable multimodal models.", "Mmmu, Yue", "Zhao et\u00a0al. (2023): Beyond hallucinations: Enhancing lvlms"]
    
    # Evaluate the dummy method
    metrics = evaluator.evaluate_method("dummy_method", dummy_method)

    # Compare methods (in this case, just the dummy method)
    evaluator.compare_methods(["dummy_method"])

    
    