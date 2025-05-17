#!/usr/bin/env python
"""
Script demonstrating the usage of citation coverage evaluation tools.

This script shows how to:
1. Evaluate citation prediction methods for their coverage of ground truth citations
2. Compare different methods based on coverage metrics at paper, section, and paragraph levels
"""

import logging
from pathlib import Path
from typing import List

from src.evaluation.coverage_evaluator import CoverageCitationEvaluator, TestTaskType

from src.methods.dummy import dummy_method
from src.methods.search_based_coverage import predict_search_based  # Replace with actual method import
from src.methods.naive_llm_based import predict_pure_llm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("arxiv").setLevel(logging.WARNING)

def run_coverage_evaluation():
    """Run coverage evaluation for citation prediction methods."""
    logger.info("Running citation coverage evaluation")
    
    # Setup paths
    test_data_path = Path("data/output/dataset_coverage")
    results_dir = Path("eval/coverage_evaluation_results_search_based")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test tasks to evaluate (paper, section, paragraph level)
    test_tasks = [
        TestTaskType.PAPER, 
        TestTaskType.SECTION,
        TestTaskType.PARAGRAPH
    ]
    
    # Initialize evaluator
    evaluator = CoverageCitationEvaluator(
        test_data_path=test_data_path,
        output_dir=results_dir,
        test_tasks=test_tasks
    )
    
    # Define methods to evaluate
    methods = {
        # "dummy": dummy_method,
        "naive_llm_based": predict_pure_llm,
        "search_based": predict_search_based,
    }
    
    # Evaluate methods
    method_metrics = {}
    for method_name, method_func in methods.items():
        logger.info(f"Evaluating method: {method_name}")
        metrics = evaluator.evaluate_method(
            method_name=method_name,
            method_func=method_func,
            n_papers=4
        )
        
        method_metrics[method_name] = metrics
        logger.info(f"Method {method_name} evaluation completed")
    
    # Compare methods
    evaluator.compare_methods(list(methods.keys()))
    logger.info(f"Coverage comparison results saved to {results_dir}")
    
    return method_metrics

if __name__ == "__main__":
    logger.info("Starting Citation Coverage Evaluation")
    
    run_coverage_evaluation()
    
    logger.info("Citation Coverage Evaluation completed successfully!")