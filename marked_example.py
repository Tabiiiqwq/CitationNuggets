#!/usr/bin/env python
"""
Example script demonstrating the usage of marked citation evaluation tools.

This script shows how to:
1. Evaluate citation prediction methods using text with [CITATION] markers
2. Compare different methods for accuracy in citation content
"""

import logging
from pathlib import Path

from src.evaluation.marked_evaluator import MarkedCitationEvaluator
from src.methods.marked_baseline import predict_random_marked_citations, predict_context_marked_citations
from src.methods.marked_ours import predict_graph_citations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_marked_evaluation():
    """Example of evaluating citation prediction methods using marked text."""
    logger.info("Running example: Marked Citation Evaluation")
    
    # Setup paths
    test_data_path = Path("data/output/dataset")
    results_dir = Path("data/output/marked_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify test data exists
    test_json_path = test_data_path / "splits/test.json"
    if not test_json_path.exists():
        logger.error("Test data not found at %s. Please run example.py first.", test_json_path)
        return
    
    # Define methods to evaluate
    methods = {
        "random": predict_random_marked_citations,
        "context": predict_context_marked_citations,
        "ours": predict_graph_citations
    }
    
    # Initialize evaluator
    evaluator = MarkedCitationEvaluator(test_data_path, results_dir)
    
    # Evaluate methods
    method_metrics = {}
    for method_name, method_func in methods.items():
        logger.info("Evaluating method: %s", method_name)
        metrics = evaluator.evaluate_method(
            method_name=method_name,
            method_func=method_func,
            parallel=False  # Set to True for faster evaluation
        )
        
        method_metrics[method_name] = metrics
        
        logger.info("Results for %s:", method_name)
        logger.info("  Citation Accuracy: %.4f", metrics["citation_accuracy"])
        logger.info("  Length Match Rate: %.4f", metrics["length_match_rate"])
    
    # Compare methods
    evaluator.compare_methods(list(methods.keys()))
    logger.info("Comparison results saved to %s", results_dir)
    
    return method_metrics

if __name__ == "__main__":
    logger.info("Marked Citation Evaluation Examples")
    
    example_marked_evaluation()
    
    logger.info("\nAll examples completed successfully!")
