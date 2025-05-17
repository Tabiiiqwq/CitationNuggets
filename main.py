#!/usr/bin/env python
"""
CLI for Citation Fill-in Dataset Construction and Evaluation Tools.

This script provides command-line interfaces for:
1. Processing academic papers to extract citations
2. Building datasets from processed papers
3. Evaluating citation prediction methods
"""

import os
import sys
import fire
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

from src.dataset_construction.html_paper_processor import PaperProcessor
from src.dataset_construction.dataset_builder_coverage import CitationDatasetBuilder
from src.evaluation.coverage_evaluator import CoverageCitationEvaluator, TestTaskType, eval_with_json_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CitationFillinCLI:
    """Command-line interface for citation fill-in dataset construction and evaluation."""
    
    def process_papers(self, 
                      input_dir: str, 
                      output_dir: str = "data/output/processed_ACL_html",
                      file_pattern: str = "*.html") -> None:
        """
        Process academic papers to extract citations and create different versions.
        
        Args:
            input_dir: Directory containing papers to process
            output_dir: Directory to save processed papers
            file_pattern: File pattern to match (e.g., *.html)
        """
        logger.info(f"Processing papers from {input_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Check if input directory exists
        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize paper processor
        processor = PaperProcessor(output_path)
        
        # Process papers
        if input_path.is_file() and str(input_path).lower().endswith(".html"):
            # Process a single paper
            processor.process_html(input_path)
            logger.info(f"Processed {input_path.name}")
        else:
            # Process all papers in the directory
            processed = processor.process_directory(input_path)
            logger.info(f"Processed {len(processed)} papers from {input_dir}")
    
    def build_dataset(self, 
                     processed_dir: str = "data/output/processed", 
                     output_dir: str = "data/output/dataset",
                     split_ratio: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                     seed: int = 42,
                     export_huggingface: bool = True) -> None:
        """
        Build citation dataset from processed papers.
        
        Args:
            processed_dir: Directory containing processed papers
            output_dir: Directory to save the dataset
            split_ratio: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
            export_huggingface: Whether to export the dataset in Hugging Face format
        """
        logger.info(f"Building dataset from {processed_dir}")
        
        processed_path = Path(processed_dir)
        output_path = Path(output_dir)
        
        # Check if processed directory exists
        if not processed_path.exists():
            logger.error(f"Processed directory {processed_dir} does not exist")
            sys.exit(1)
        
        # Create dataset builder
        builder = CitationDatasetBuilder(processed_path, output_path)
        
        # Build dataset
        splits = builder.build_dataset(split_ratio=split_ratio, seed=seed)
        
        # Export to Hugging Face format
        if export_huggingface:
            builder.export_huggingface_format()
            logger.info("Exported dataset to Hugging Face format")
        
        logger.info(f"Dataset built with {sum(len(df) for df in splits.values())} papers")
        logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    def evaluate_coverage_json(self,
                             json_path: str,
                             output_dir: str,
                             test_data_path: str,
                             method_name: str = "json_method",
                             task_levels: List[str] = None) -> None:
        """
        Evaluate citation predictions using a JSON file of predictions.
        
        Args:
            json_path: Path to the JSON file containing predictions
            output_dir: Directory to save evaluation results
            test_data_path: Path to test data
            method_name: Name of the method (for reporting)
            task_levels: Levels to evaluate (PAPER, SECTION, PARAGRAPH)
        """
        logger.info(f"Evaluating coverage from JSON file: {json_path}")
        
        # Convert string task levels to TestTaskType enum values
        if task_levels is None:
            task_levels = ["PAPER", "SECTION", "PARAGRAPH"]
            
        test_tasks = [TestTaskType[level] for level in task_levels]
        
        # Run evaluation
        metrics = eval_with_json_result(
            json_path=json_path,
            output_dir=output_dir,
            test_data_path=test_data_path,
            test_tasks=test_tasks,
            method_name=method_name
        )
        
        # Print key metrics
        print("\n===== Coverage Evaluation Results =====")
        if "paper_coverage" in metrics:
            print(f"Paper Coverage: {metrics['paper_coverage']:.4f}")
        if "section_coverage" in metrics:
            print(f"Section Coverage: {metrics['section_coverage']:.4f}")
        if "paragraph_coverage" in metrics:
            print(f"Paragraph Coverage: {metrics['paragraph_coverage']:.4f}")
        
        logger.info(f"Evaluation results saved to {Path(output_dir) / 'coverage_results' / method_name}")
    
    def run_coverage_evaluation(self,
                              test_data_path: str,
                              output_dir: str,
                              methods: List[str] = None,
                              task_levels: List[str] = None,
                              n_papers: int = None) -> None:
        """
        Run coverage evaluation for one or more citation prediction methods.
        
        Args:
            test_data_path: Path to test data
            output_dir: Directory to save evaluation results
            methods: List of method names to evaluate
            task_levels: Levels to evaluate (PAPER, SECTION, PARAGRAPH)
            n_papers: Number of papers to evaluate (None for all)
        """
        logger.info(f"Running citation coverage evaluation")
        
        # Set default methods if none provided
        if methods is None:
            methods = ["dummy"]
        
        # Convert string task levels to TestTaskType enum values
        if task_levels is None:
            task_levels = ["PAPER", "SECTION", "PARAGRAPH"]
            
        test_tasks = [TestTaskType[level] for level in task_levels]
        
        # Initialize evaluator
        results_path = Path(output_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        evaluator = CoverageCitationEvaluator(
            test_data_path=test_data_path,
            output_dir=results_path,
            test_tasks=test_tasks
        )
        
        # Import methods dynamically
        method_functions = {}
        
        if "dummy" in methods:
            from src.methods.dummy import dummy_method
            method_functions["dummy"] = dummy_method
            
        if "search_based" in methods:
            from src.methods.search_based_coverage import predict_search_based
            method_functions["search_based"] = predict_search_based
            
        if "naive_llm_based" in methods:
            from src.methods.naive_llm_based import predict_pure_llm
            method_functions["naive_llm_based"] = predict_pure_llm
        
        # Evaluate methods
        for method_name, method_func in method_functions.items():
            logger.info(f"Evaluating method: {method_name}")
            metrics = evaluator.evaluate_method(
                method_name=method_name,
                method_func=method_func,
                n_papers=n_papers
            )
            
            logger.info(f"Method {method_name} evaluation completed")
        
        # Compare methods
        evaluator.compare_methods(list(method_functions.keys()))
        
        logger.info(f"Coverage comparison results saved to {results_path}")

if __name__ == "__main__":
    fire.Fire(CitationFillinCLI)
