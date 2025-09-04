"""
Data loading and dataset classes for k-fold training
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class LabelEncoder:
    """Label encoder that returns None for unknown labels - matches granular_control"""
    
    def __init__(self, labels=None):
        """Initialize encoder from list of labels"""
        if labels:
            unique_labels = sorted(set(labels))
            self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    @classmethod
    def from_dict(cls, encoder_dict: Dict) -> 'LabelEncoder':
        """Reconstruct from saved dictionary"""
        encoder = cls()
        encoder.label_to_index = encoder_dict['label_to_index']
        encoder.index_to_label = {int(k): v for k, v in encoder_dict['index_to_label'].items()}
        return encoder
    
    def encode(self, label: str) -> Optional[int]:
        """Encode label to index, returns None for unknown"""
        return self.label_to_index.get(label, None)
    
    def decode(self, index: int) -> Optional[str]:
        """Decode index to label"""
        return self.index_to_label.get(index, None)


class FungalSequenceDataset(Dataset):
    """Dataset for fungal sequence classification (single-level)"""
    
    def __init__(self, sequences: List[str], labels: List[str], 
                 tokenizer, max_length: int = 512, label_encoder: Optional[LabelEncoder] = None):
        """
        Args:
            sequences: List of DNA sequences
            labels: List of species labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            label_encoder: Optional pre-built label encoder
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use provided encoder or create new one
        if label_encoder:
            self.label_encoder = label_encoder
            self.label_to_idx = label_encoder.label_to_index
            self.idx_to_label = label_encoder.index_to_label
            self.num_classes = len(self.label_to_idx)
        else:
            # Fallback: create from data (old behavior)
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.num_classes = len(unique_labels)
            # Create encoder object for consistency
            self.label_encoder = LabelEncoder(labels)
        
        # Count how many samples would be unknown
        if label_encoder:
            unknown_count = sum(1 for label in labels if label_encoder.encode(label) is None)
            if unknown_count > 0:
                logger.warning(f"Dataset has {unknown_count} samples with unknown labels")
        
        logger.info(f"Dataset created with {len(sequences)} sequences, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Check if label is known
        label_idx = self.label_encoder.encode(label)
        if label_idx is None:
            # Unknown label - return None to filter out
            return None
        
        # Tokenize sequence
        encoding = self.tokenizer.encode(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label_idx, dtype=torch.long)
        }


class HierarchicalFungalDataset(Dataset):
    """Dataset for hierarchical fungal sequence classification"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512,
                 taxonomic_levels: List[str] = None, label_encoders: Optional[Dict[str, LabelEncoder]] = None):
        """
        Args:
            df: DataFrame with sequences and hierarchical labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            taxonomic_levels: List of taxonomic levels to use
            label_encoders: Optional pre-built label encoders for each level
        """
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default taxonomic levels
        if taxonomic_levels is None:
            taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        self.taxonomic_levels = taxonomic_levels
        
        # Use provided encoders or create new ones
        if label_encoders:
            # Use pre-built encoders
            self.label_encoders = label_encoders
            self.num_classes_per_level = []
            for level in self.taxonomic_levels:
                if level not in label_encoders:
                    raise ValueError(f"Missing label encoder for level: {level}")
                self.num_classes_per_level.append(len(label_encoders[level].label_to_index))
        else:
            # Fallback: create from data (old behavior)
            self.label_encoders = {}
            self.num_classes_per_level = []
            
            for level in self.taxonomic_levels:
                if level not in df.columns:
                    # Try to create genus_species column if species level is requested
                    if level == 'species' and 'genus' in df.columns and 'species' in df.columns:
                        df['species'] = df['genus'] + '_' + df['species']
                    else:
                        raise ValueError(f"Taxonomic level '{level}' not found in data")
                
                # Get unique labels for this level
                unique_labels = sorted(df[level].dropna().unique())
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                idx_to_label = {idx: label for label, idx in label_to_idx.items()}
                
                self.label_encoders[level] = {
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label,
                    'num_classes': len(unique_labels)
                }
                self.num_classes_per_level.append(len(unique_labels))
        
        # Filter out rows with missing sequences or labels
        for level in self.taxonomic_levels:
            self.df = self.df[self.df[level].notna()]
        self.df = self.df[self.df['sequence'].notna()].reset_index(drop=True)
        
        logger.info(f"Hierarchical dataset: {len(self.df)} sequences")
        for level in self.taxonomic_levels:
            if isinstance(self.label_encoders[level], LabelEncoder):
                num_classes = len(self.label_encoders[level].label_to_index)
            else:
                # Old dict format
                num_classes = self.label_encoders[level]['num_classes']
            logger.info(f"  {level}: {num_classes} classes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['sequence']
        
        # Tokenize sequence
        encoding = self.tokenizer.encode(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Prepare hierarchical labels
        labels = {}
        for level in self.taxonomic_levels:
            label = row[level]
            
            # Handle both new LabelEncoder objects and old dict format
            if isinstance(self.label_encoders[level], LabelEncoder):
                label_idx = self.label_encoders[level].encode(label)
                if label_idx is None:
                    # Unknown label - skip this sample
                    return None
            else:
                # Old dict format (fallback)
                label_idx = self.label_encoders[level]['label_to_idx'].get(label, None)
                if label_idx is None:
                    return None
            
            labels[level] = {
                'label': label,
                'encoded_label': label_idx
            }
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'output': labels
        }


def load_fold_data(data_path: Path, fold: int, fold_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split data for a specific fold
    
    Args:
        data_path: Path to the CSV file with all data
        fold: Fold number (1-10)
        fold_type: "exp1_sequence_fold" or "exp2_species_fold"
        
    Returns:
        train_df, val_df
    """
    # Read data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} sequences from {data_path}")
    
    # Determine fold column
    fold_column = 'fold_exp1' if 'exp1' in fold_type else 'fold_exp2'
    
    if fold_column not in df.columns:
        raise ValueError(f"Fold column '{fold_column}' not found in data")
    
    # Split into train and validation
    val_df = df[df[fold_column] == fold].copy()
    train_df = df[df[fold_column] != fold].copy()
    
    logger.info(f"Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")
    
    return train_df, val_df


def prepare_data_for_training(df: pd.DataFrame) -> Dict:
    """
    Prepare dataframe for training
    
    Args:
        df: DataFrame with sequences and labels
        
    Returns:
        Dictionary with sequences, labels, and metadata
    """
    # Required columns
    required_cols = ['sequence', 'genus', 'species']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Use species labels directly (to match encoder format)
    df['label'] = df['species']
    
    # Remove rows with missing data
    df = df.dropna(subset=['sequence', 'label'])
    
    return {
        'sequences': df['sequence'].tolist(),
        'labels': df['label'].tolist(),
        'num_sequences': len(df),
        'num_classes': df['label'].nunique()
    }


def collate_fn_filter_none(batch):
    """Custom collate function that filters out None samples"""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def create_data_loaders(train_data, val_data, tokenizer, 
                       batch_size: int = 32, max_length: int = 512,
                       num_workers: int = 4, hierarchical: bool = False,
                       taxonomic_levels: List[str] = None, 
                       label_encoder: Optional[LabelEncoder] = None,
                       label_encoders: Optional[Dict[str, LabelEncoder]] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation
    
    Args:
        train_data: Training data (Dict for single, DataFrame for hierarchical)
        val_data: Validation data (Dict for single, DataFrame for hierarchical)
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        hierarchical: Whether to use hierarchical classification
        taxonomic_levels: List of taxonomic levels for hierarchical
        label_encoder: Single LabelEncoder for single-level classification
        label_encoders: Dict of LabelEncoders for hierarchical classification
        
    Returns:
        train_loader, val_loader
    """
    if hierarchical:
        # Create hierarchical datasets
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("For hierarchical classification, data must be a DataFrame")
        
        train_dataset = HierarchicalFungalDataset(
            train_data,
            tokenizer,
            max_length,
            taxonomic_levels,
            label_encoders=label_encoders  # Pass pre-built encoders
        )
        
        val_dataset = HierarchicalFungalDataset(
            val_data,
            tokenizer,
            max_length,
            taxonomic_levels,
            label_encoders=label_encoders  # Use same encoders for validation
        )
        
        # No need to copy encoders if using pre-built ones
    else:
        # Create single-level datasets
        if isinstance(train_data, pd.DataFrame):
            # Convert DataFrame to dict format for single-level
            train_data = prepare_data_for_training(train_data)
            val_data = prepare_data_for_training(val_data)
        
        train_dataset = FungalSequenceDataset(
            train_data['sequences'],
            train_data['labels'],
            tokenizer,
            max_length,
            label_encoder=label_encoder  # Pass single pre-built encoder
        )
        
        val_dataset = FungalSequenceDataset(
            val_data['sequences'],
            val_data['labels'],
            tokenizer,
            max_length,
            label_encoder=label_encoder  # Use same encoder for validation
        )
    
    # Create loaders with collate function to filter None samples
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_filter_none  # Filter None samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_filter_none  # Filter None samples
    )
    
    return train_loader, val_loader