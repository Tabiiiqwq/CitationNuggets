"""
Module for evaluating citation prediction performance.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CitationEvaluator:
    """
    Class for evaluating citation prediction performance.
    Provides metrics for citation coverage and placement accuracy.
    """
    
    def __init__(self, ground_truth_path: Union[str, Path]):
        """
        Initialize the CitationEvaluator.
        
        Args:
            ground_truth_path: Path to ground truth data (JSON file)
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth = self._load_ground_truth()
    
    def _load_ground_truth(self) -> Dict[str, Any]:
        """
        Load ground truth data from JSON file.
        
        Returns:
            Dictionary containing ground truth data
        """
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def evaluate(self, predictions_path: Union[str, Path]) -> Dict[str, float]:
        """
        Evaluate citation predictions against ground truth.
        
        Args:
            predictions_path: Path to predictions data (JSON file)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load predictions
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Calculate metrics
        coverage_metrics = self._evaluate_citation_coverage(predictions)
        placement_metrics = self._evaluate_citation_placement(predictions)
        
        # Combine metrics
        metrics = {**coverage_metrics, **placement_metrics}
        
        # Log results
        self._log_evaluation_results(metrics)
        
        return metrics
    
    def _evaluate_citation_coverage(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate citation coverage (how many ground truth citations were predicted).
        
        Args:
            predictions: Dictionary containing prediction data
            
        Returns:
            Dictionary containing coverage metrics
        """
        # Extract citation content from ground truth and predictions
        gt_citations = set(self._extract_citation_content(self.ground_truth))
        pred_citations = set(self._extract_citation_content(predictions))
        
        # Calculate metrics
        true_positives = len(gt_citations.intersection(pred_citations))
        false_positives = len(pred_citations - gt_citations)
        false_negatives = len(gt_citations - pred_citations)
        
        # Precision, Recall, F1 Score
        precision = true_positives / max(1, (true_positives + false_positives))
        recall = true_positives / max(1, (true_positives + false_negatives))
        f1 = 2 * precision * recall / max(1e-6, (precision + recall))
        
        return {
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": f1,
            "citation_coverage": recall,  # Alias for recall
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
        
        # If "references" or other formats exist, add handling here
        
        # Default: empty list if no citations found
        logger.warning("No citation content found in data")
        return []
    
    def _evaluate_citation_placement(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate citation placement accuracy.
        
        Args:
            predictions: Dictionary containing prediction data
            
        Returns:
            Dictionary containing placement accuracy metrics
        """
        # Extract citation positions from ground truth and predictions
        gt_positions = self._extract_citation_positions(self.ground_truth)
        pred_positions = self._extract_citation_positions(predictions)
        
        # Calculate position-based metrics
        position_precision, position_recall, position_f1 = self._calculate_position_metrics(gt_positions, pred_positions)
        
        # Calculate average position error
        avg_position_error = self._calculate_position_error(gt_positions, pred_positions)
        
        return {
            "position_precision": position_precision,
            "position_recall": position_recall,
            "position_f1": position_f1,
            "avg_position_error": avg_position_error,
        }
    
    def _extract_citation_positions(self, data: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Extract citation positions from data.
        
        Args:
            data: Dictionary containing citation data
            
        Returns:
            List of position tuples (start, end)
        """
        if "citations" in data and "positions" in data["citations"]:
            return data["citations"]["positions"]
        
        # Handle different formats
        if "citation_info" in data and "positions" in data["citation_info"]:
            return data["citation_info"]["positions"]
        
        # Default: empty list if no positions found
        logger.warning("No citation positions found in data")
        return []
    
    def _calculate_position_metrics(self, gt_positions: List[Tuple[int, int]], 
                                   pred_positions: List[Tuple[int, int]]) -> Tuple[float, float, float]:
        """
        Calculate position-based precision, recall, and F1 score.
        
        A predicted position is considered correct if it overlaps with a 
        ground truth position or is within a small distance threshold.
        
        Args:
            gt_positions: List of ground truth position tuples
            pred_positions: List of predicted position tuples
            
        Returns:
            Tuple of (precision, recall, F1 score)
        """
        # Define overlap threshold (consider positions matching if they are within this many characters)
        position_threshold = 50  # Characters
        
        # Count matches
        true_positives = 0
        
        # Track ground truth positions that have been matched
        matched_gt_positions = set()
        
        for pred_start, pred_end in pred_positions:
            # Check if this prediction matches any ground truth position
            for i, (gt_start, gt_end) in enumerate(gt_positions):
                # Check if positions overlap or are close
                if (max(pred_start, gt_start) <= min(pred_end, gt_end) or 
                    abs(pred_start - gt_start) <= position_threshold or 
                    abs(pred_end - gt_end) <= position_threshold):
                    
                    # Found a match
                    true_positives += 1
                    matched_gt_positions.add(i)
                    break
        
        # Calculate metrics
        precision = true_positives / max(1, len(pred_positions))
        recall = len(matched_gt_positions) / max(1, len(gt_positions))
        f1 = 2 * precision * recall / max(1e-6, (precision + recall))
        
        return precision, recall, f1
    
    def _calculate_position_error(self, gt_positions: List[Tuple[int, int]], 
                                 pred_positions: List[Tuple[int, int]]) -> float:
        """
        Calculate average position error between matched positions.
        
        Args:
            gt_positions: List of ground truth position tuples
            pred_positions: List of predicted position tuples
            
        Returns:
            Average position error in characters
        """
        if not gt_positions or not pred_positions:
            return float('inf')
        
        # Match predictions to closest ground truth positions
        total_error = 0
        matched_count = 0
        
        for pred_start, pred_end in pred_positions:
            # Find closest ground truth position
            min_error = float('inf')
            
            for gt_start, gt_end in gt_positions:
                # Calculate error as average of start and end differences
                error = (abs(pred_start - gt_start) + abs(pred_end - gt_end)) / 2
                min_error = min(min_error, error)
            
            if min_error < float('inf'):
                total_error += min_error
                matched_count += 1
        
        # Return average error
        return total_error / max(1, matched_count)
    
    def _log_evaluation_results(self, metrics: Dict[str, float]) -> None:
        """
        Log evaluation results.
        
        Args:
            metrics: Dictionary containing evaluation metrics
        """
        logger.info("===== Citation Evaluation Results =====")
        logger.info(f"Citation Coverage (Recall): {metrics['citation_recall']:.4f}")
        logger.info(f"Citation Precision: {metrics['citation_precision']:.4f}")
        logger.info(f"Citation F1 Score: {metrics['citation_f1']:.4f}")
        logger.info(f"Position Precision: {metrics['position_precision']:.4f}")
        logger.info(f"Position Recall: {metrics['position_recall']:.4f}")
        logger.info(f"Position F1 Score: {metrics['position_f1']:.4f}")
        logger.info(f"Average Position Error: {metrics['avg_position_error']:.2f} characters")
