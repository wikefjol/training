"""
Data loading and dataset classes for k-fold training
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FungalSequenceDataset(Dataset):
    """Dataset for fungal sequence classification (single-level)"""
    
    def __init__(self, sequences: List[str], labels: List[str], 
                 tokenizer, max_length: int = 512):
        """
        Args:
            sequences: List of DNA sequences
            labels: List of species labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        logger.info(f"Dataset created with {len(sequences)} sequences, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
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
            'label': torch.tensor(self.label_to_idx[label], dtype=torch.long)
        }


class HierarchicalFungalDataset(Dataset):
    """Dataset for hierarchical fungal sequence classification"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512,
                 taxonomic_levels: List[str] = None):
        """
        Args:
            df: DataFrame with sequences and hierarchical labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            taxonomic_levels: List of taxonomic levels to use
        """
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default taxonomic levels
        if taxonomic_levels is None:
            taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        self.taxonomic_levels = taxonomic_levels
        
        # Create label encoders for each level
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
            logger.info(f"  {level}: {self.label_encoders[level]['num_classes']} classes")
    
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
            label_idx = self.label_encoders[level]['label_to_idx'][label]
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
    
    # Create species-level labels
    df['label'] = df['genus'] + '_' + df['species']
    
    # Remove rows with missing data
    df = df.dropna(subset=['sequence', 'label'])
    
    return {
        'sequences': df['sequence'].tolist(),
        'labels': df['label'].tolist(),
        'num_sequences': len(df),
        'num_classes': df['label'].nunique()
    }


def create_data_loaders(train_data, val_data, tokenizer, 
                       batch_size: int = 32, max_length: int = 512,
                       num_workers: int = 4, hierarchical: bool = False,
                       taxonomic_levels: List[str] = None) -> Tuple[DataLoader, DataLoader]:
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
            taxonomic_levels
        )
        
        val_dataset = HierarchicalFungalDataset(
            val_data,
            tokenizer,
            max_length,
            taxonomic_levels
        )
        
        # Ensure validation uses same label mappings as training
        val_dataset.label_encoders = train_dataset.label_encoders
        val_dataset.num_classes_per_level = train_dataset.num_classes_per_level
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
            max_length
        )
        
        val_dataset = FungalSequenceDataset(
            val_data['sequences'],
            val_data['labels'],
            tokenizer,
            max_length
        )
        
        # Ensure validation uses same label mapping as training
        val_dataset.label_to_idx = train_dataset.label_to_idx
        val_dataset.idx_to_label = train_dataset.idx_to_label
        val_dataset.num_classes = train_dataset.num_classes
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader