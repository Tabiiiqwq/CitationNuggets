"""
Module for building citation datasets from processed papers.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import pandas as pd
from tqdm import tqdm

from .paper_processor import PaperProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CitationDatasetBuilder:
    """
    Class for building citation datasets from processed papers.
    This class handles organizing processed papers into train/val/test splits
    and provides functionality to export the dataset in various formats.
    """
    
    def __init__(self, processed_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the CitationDatasetBuilder.
        
        Args:
            processed_dir: Directory containing processed papers
            output_dir: Directory to save the dataset
        """
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories in processed_dir
        self.full_paper_dir = self.processed_dir / "full_paper"
        self.paragraph_dir = self.processed_dir / "paragraph" 
        self.section_dir = self.processed_dir / "section"
        
        # Verify directories exist
        for directory in [self.full_paper_dir, self.paragraph_dir, self.section_dir]:
            if not directory.exists():
                raise ValueError(f"Directory {directory} does not exist")
    
    def build_dataset(self, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                      seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Build the citation dataset with train/val/test splits.
        
        Args:
            split_ratio: Tuple of (train, val, test) ratios
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing DataFrames for train, val, and test splits
        """
        # Validate split ratio
        if sum(split_ratio) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Set random seed
        random.seed(seed)
        
        # Get all paper IDs (filenames without extension)
        paper_ids = [f.stem for f in self.full_paper_dir.glob("*.json")]
        logger.info(f"Found {len(paper_ids)} processed papers")
        
        # Shuffle and split
        random.shuffle(paper_ids)
        train_size = int(len(paper_ids) * split_ratio[0])
        val_size = int(len(paper_ids) * split_ratio[1])
        
        train_ids = paper_ids[:train_size]
        val_ids = paper_ids[train_size:train_size + val_size]
        test_ids = paper_ids[train_size + val_size:]
        
        logger.info(f"Split sizes: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        
        # Build DataFrames for each split
        train_df = self._build_split_dataframe(train_ids, "train")
        val_df = self._build_split_dataframe(val_ids, "val")
        test_df = self._build_split_dataframe(test_ids, "test")
        
        # Save splits
        self._save_splits({"train": train_df, "val": val_df, "test": test_df})
        
        return {"train": train_df, "val": val_df, "test": test_df}
    
    def _build_split_dataframe(self, paper_ids: List[str], split_name: str) -> pd.DataFrame:
        """
        Build a DataFrame for a specific split.
        
        Args:
            paper_ids: List of paper IDs to include in the split
            split_name: Name of the split (train, val, test)
            
        Returns:
            DataFrame containing data for the split
        """
        data = []
        
        for paper_id in tqdm(paper_ids, desc=f"Building {split_name} split"):
            try:
                with open(self.full_paper_dir / f"{paper_id}.json", "r", encoding="utf-8") as f:
                    full_paper_data = json.load(f)

                with open(self.paragraph_dir / f"{paper_id}.json", "r", encoding="utf-8") as f:
                    paragraphs_data = json.load(f)

                with open(self.section_dir / f"{paper_id}.json", "r", encoding="utf-8") as f:
                    sections_data = json.load(f)
                
                # Add to data list
                data.append({
                    "paper_id": paper_id,
                    "full_paper": full_paper_data,
                    "paragraphs": paragraphs_data,
                    "sections": sections_data,
                })
            except Exception as e:
                logger.error(f"Error processing {paper_id}: {str(e)}")
        
        # Create DataFrame
        
        df = pd.DataFrame(data)
        logger.info(f"Created {split_name} DataFrame with {len(df)} rows")
        
        return df
    
    def _save_splits(self, splits: Dict[str, pd.DataFrame]) -> None:
        """
        Save dataset splits to disk.
        
        Args:
            splits: Dictionary of split DataFrames
        """
        # Create splits directory
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save each split
        for split_name, df in splits.items():
            # Check if DataFrame is not empty and contains the paper_id column
            if not df.empty and 'paper_id' in df.columns:
                # Save as CSV
                df[["paper_id"]].to_csv(splits_dir / f"{split_name}_papers.csv", index=False)
            else:
                # Create empty CSV with just the header
                with open(splits_dir / f"{split_name}_papers.csv", "w") as f:
                    f.write("paper_id\n")
            
            # Save as JSON (works even with empty DataFrames)
            df.to_json(splits_dir / f"{split_name}.json", orient="records", indent=2)
        
        # Save metadata
        metadata = {
            "dataset_size": sum(len(df) for df in splits.values()),
            "split_sizes": {split: len(df) for split, df in splits.items()}
        }
        
        with open(self.output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved dataset splits to {splits_dir}")
    
    def export_huggingface_format(self) -> None:
        """
        Export the dataset in a format compatible with Hugging Face datasets.
        """
        # Create huggingface directory
        hf_dir = self.output_dir / "huggingface"
        hf_dir.mkdir(exist_ok=True)
        
        splits_dir = self.output_dir / "splits"
        for split in ["train", "val", "test"]:
            try:
                # Check if the split JSON file exists
                split_file = splits_dir / f"{split}.json"
                if not split_file.exists():
                    logger.warning(f"Split file {split_file} does not exist, creating empty file")
                    # Create empty JSONL file
                    with open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                        pass
                    continue
                
                # Load the split - handle empty files
                try:
                    df = pd.read_json(split_file, orient="records")
                except ValueError as e:
                    if "Unexpected end of file" in str(e):
                        # Empty JSON file or invalid JSON
                        logger.warning(f"Empty or invalid JSON file for {split} split")
                        with open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                            pass
                        continue
                    else:
                        raise
                
                # Handle empty DataFrame
                if df.empty:
                    logger.warning(f"Empty DataFrame for {split} split")
                    with open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                        pass
                    continue
                
                # Process the data
                processed_data = []
                for _, row in df.iterrows():
                    # Check if required columns exist
                    if all(col in row for col in ["paper_id", "full_paper", "paragraphs", "sections"]):
                        processed_data.append({
                            "paper_id": row["paper_id"],
                            "full_paper": row["full_paper"],  # Papers with citations removed
                            "paragraphs": row["paragraphs"],  # Papers with citation markers
                            "sections": row["sections"]  # Citation information
                        })
                    else:
                        logger.warning(f"Missing required columns in row, skipping: {row.name}")
                
                # Save as jsonl (one JSON object per line)
                with open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                    for item in processed_data:
                        f.write(json.dumps(item) + "\n")
                
                logger.info(f"Exported {split} split with {len(processed_data)} entries to {hf_dir / f'{split}.jsonl'}")
            except Exception as e:
                logger.error(f"Error exporting {split} split: {str(e)}")
                # Create empty file on error
                with open(hf_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                    pass



if __name__ == '__main__':
    processed_path = r'data\output\processed_ACL_html'
    output_path = r'data\output\dataset_coverage'


    
    builder = CitationDatasetBuilder(processed_path, output_path)
        
    # Build dataset
    splits = builder.build_dataset(split_ratio=(0, 0, 1))
    
    # Export to Hugging Face format
    builder.export_huggingface_format()
    logger.info("Exported dataset to Hugging Face format")
    
    logger.info(f"Dataset built with {sum(len(df) for df in splits.values())} papers")
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")