#!/usr/bin/env python
"""
Example script demonstrating the usage of Citation Fill-in tools.

This script shows how to:
1. Process papers to extract citations
2. Build datasets from processed papers
3. Evaluate citation prediction methods
"""

import os
import sys
import logging
from pathlib import Path

from src.dataset_construction.paper_processor import PaperProcessor
from src.dataset_construction.dataset_builder import CitationDatasetBuilder
from src.evaluation.evaluator import CitationMethodEvaluator
from src.methods.baseline import RandomCitationPredictor, predict_random_citations
from src.methods.llm_based import predict_with_openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_process_papers():
    """Example of processing papers to extract citations."""
    logger.info("Running example: Process papers")
    
    # Create mock paper with citations for demo
    mock_paper_dir = Path("data/input/mock")
    mock_paper_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mock paper with citations
    mock_paper_path = mock_paper_dir / "mock_paper.txt"
    mock_paper_content = """
TITLE: Advances in Natural Language Processing
ABSTRACT: 
This paper reviews recent advances in Natural Language Processing (NLP). 
We discuss transformer-based models that have revolutionized the field [Smith et al., 2020].
Several benchmarks show the superiority of these approaches [1, 2, 3].

INTRODUCTION:
Natural Language Processing has seen tremendous progress in recent years (Johnson and Williams, 2019).
The introduction of attention mechanisms [4] has led to significant improvements.
These developments build upon earlier work on recurrent neural networks [Chen et al., 2018].

METHODOLOGY:
Our approach extends previous methods [Wang et al., 2017] by incorporating...
As shown by [Patel, 2021], this technique improves performance on several benchmarks.

RESULTS:
The results, consistent with [Kumar and Lee, 2022], demonstrate significant improvements...
Our method outperforms previous approaches [7, 8] by a substantial margin.

CONCLUSION:
We have presented a novel approach for NLP tasks. 
Future work could explore extensions to other domains [Brown et al., 2016].
    """
    
    with open(mock_paper_path, "w", encoding="utf-8") as f:
        f.write(mock_paper_content)
    
    # Initialize paper processor
    output_dir = Path("data/output/processed")
    processor = PaperProcessor(output_dir)
    
    # Process the mock paper
    result = processor.process_pdf(mock_paper_path)
    
    logger.info(f"Processed mock paper: {mock_paper_path}")
    logger.info(f"Found {len(result['citations']['content'])} citations")
    logger.info(f"Output saved to: {output_dir}")
    
    # Display sample of processed content
    logger.info("\nSample of paper with NO citations:")
    print(result["no_citations_text"][:300] + "...\n")
    
    logger.info("Sample of paper with MARKED citations:")
    print(result["marked_citations_text"][:300] + "...\n")
    
    logger.info("Citation information:")
    citation_positions = list(zip(
        result["citations"]["content"], 
        result["citations"]["positions"]
    ))
    for i, (citation, pos) in enumerate(citation_positions):
        print(f"  {i+1}. {citation} (position: {pos})")

def example_build_dataset():
    """Example of building a dataset from processed papers."""
    logger.info("\nRunning example: Build dataset")
    
    # Initialize dataset builder
    processed_dir = Path("data/output/processed")
    output_dir = Path("data/output/dataset")
    
    builder = CitationDatasetBuilder(processed_dir, output_dir)
    
    # Build dataset
    splits = builder.build_dataset(split_ratio=(0.7, 0.15, 0.15), seed=42)
    
    # Export to Hugging Face format
    builder.export_huggingface_format()
    
    logger.info(f"Dataset built with {sum(len(df) for df in splits.values())} papers")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    logger.info(f"Dataset saved to: {output_dir}")

def example_baseline_prediction():
    """Example of using baseline prediction methods."""
    logger.info("\nRunning example: Baseline prediction")
    
    # Load a sample paper without citations
    processed_dir = Path("data/output/processed")
    no_citations_dir = processed_dir / "no_citations"
    
    # Get first available paper
    sample_papers = list(no_citations_dir.glob("*.txt"))
    if not sample_papers:
        logger.error("No processed papers found. Run example_process_papers first.")
        return
    
    sample_paper_path = sample_papers[0]
    with open(sample_paper_path, "r", encoding="utf-8") as f:
        paper_text = f.read()
    
    logger.info(f"Using sample paper: {sample_paper_path}")
    
    # Initialize random predictor
    predictor = RandomCitationPredictor()
    
    # Make predictions
    predictions = predictor.predict(paper_text)
    
    logger.info(f"Made {len(predictions['citations']['content'])} citation predictions")
    logger.info("Sample predictions:")
    # Convert zip to list before slicing
    citation_positions = list(zip(
        predictions["citations"]["content"], 
        predictions["citations"]["positions"]
    ))
    for i, (citation, pos) in enumerate(citation_positions[:5]):  # Show first 5 predictions
        # Get context around citation position
        context_start = max(0, pos-20)
        context_end = min(len(paper_text), pos+20)
        context = paper_text[context_start:context_end]
        print(f"  {i+1}. {citation} at position {pos}")
        print(f"     Context: \"...{context}...\"")

def example_evaluation():
    """Example of evaluating citation prediction methods."""
    logger.info("\nRunning example: Evaluation")
    
    # Setup paths
    test_data_path = Path("data/output/dataset")
    results_dir = Path("data/output/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify test data exists
    test_json_path = test_data_path / "splits/test.json"
    if not test_json_path.exists():
        logger.error("Test data not found. Run example_build_dataset first.")
        return
    
    # Define example prediction methods
    methods = {
        "random": predict_random_citations,
        # Uncomment to test with OpenAI (requires API key)
        # "openai": predict_with_openai,
    }
    
    # Initialize evaluator
    evaluator = CitationMethodEvaluator(test_data_path, results_dir)
    
    # Evaluate methods
    for method_name, method_func in methods.items():
        logger.info(f"Evaluating method: {method_name}")
        metrics = evaluator.evaluate_method(
            method_name=method_name,
            method_func=method_func,
            parallel=False  # Set to True for faster evaluation
        )
        
        logger.info(f"Results for {method_name}:")
        logger.info(f"  Citation F1: {metrics['citation_f1']:.4f}")
        logger.info(f"  Position F1: {metrics['position_f1']:.4f}")
    
    # Compare methods
    evaluator.compare_methods(list(methods.keys()))
    logger.info(f"Comparison results saved to {results_dir}")

def main():
    """Run all examples."""
    logger.info("Citation Fill-in Tools Examples")
    
    example_process_papers()
    example_build_dataset()
    example_baseline_prediction()
    example_evaluation()
    
    logger.info("\nAll examples completed successfully!")
    logger.info("For more details, check the CLI interface in main.py")

if __name__ == "__main__":
    main()
