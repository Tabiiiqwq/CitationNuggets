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

from src.dataset_construction.paper_processor import PaperProcessor
from src.dataset_construction.dataset_builder import CitationDatasetBuilder
from src.evaluation.metrics import CitationEvaluator
from src.evaluation.evaluator import CitationMethodEvaluator

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
                      output_dir: str = "data/output/processed",
                      file_pattern: str = "*.pdf") -> None:
        """
        Process academic papers to extract citations and create different versions.
        
        Args:
            input_dir: Directory containing papers to process
            output_dir: Directory to save processed papers
            file_pattern: File pattern to match (e.g., *.pdf)
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
        if input_path.is_file() and str(input_path).lower().endswith(".pdf"):
            # Process a single paper
            processor.process_pdf(input_path)
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
    
    def evaluate_paper(self, 
                     ground_truth_path: str, 
                     predictions_path: str) -> None:
        """
        Evaluate citation predictions for a single paper.
        
        Args:
            ground_truth_path: Path to ground truth data JSON file
            predictions_path: Path to predictions data JSON file
        """
        logger.info(f"Evaluating paper predictions")
        
        # Initialize evaluator
        evaluator = CitationEvaluator(ground_truth_path)
        
        # Evaluate predictions
        metrics = evaluator.evaluate(predictions_path)
        
        # Print metrics
        print("\n===== Evaluation Results =====")
        print(f"Citation Coverage (Recall): {metrics['citation_recall']:.4f}")
        print(f"Citation Precision: {metrics['citation_precision']:.4f}")
        print(f"Citation F1 Score: {metrics['citation_f1']:.4f}")
        print(f"Position Precision: {metrics['position_precision']:.4f}")
        print(f"Position Recall: {metrics['position_recall']:.4f}")
        print(f"Position F1 Score: {metrics['position_f1']:.4f}")
        print(f"Average Position Error: {metrics['avg_position_error']:.2f} characters")
    
    def compare_methods(self, 
                       test_data_path: str, 
                       results_dir: str, 
                       method_names: List[str]) -> None:
        """
        Compare multiple citation prediction methods.
        
        Args:
            test_data_path: Path to test data directory
            results_dir: Directory containing method results
            method_names: List of method names to compare
        """
        logger.info(f"Comparing methods: {', '.join(method_names)}")
        
        # Initialize evaluator
        evaluator = CitationMethodEvaluator(test_data_path, results_dir)
        
        # Compare methods
        evaluator.compare_methods(method_names)
        
        logger.info(f"Comparison results saved to {Path(results_dir) / 'method_comparison.csv'}")
        logger.info(f"Comparison plots saved to {Path(results_dir) / 'plots'}")

if __name__ == "__main__":
    fire.Fire(CitationFillinCLI)
