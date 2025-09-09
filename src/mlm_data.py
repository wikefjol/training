"""
MLM (Masked Language Modeling) dataset and data loading for pretraining
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Import augmentation function
try:
    from preprocessing import augment_sequence
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling pretraining"""
    
    def __init__(self, sequences: List[str], tokenizer, max_length: int = 512,
                 masking_prob: float = 0.15, replace_prob: float = 0.8, 
                 random_prob: float = 0.1, ignore_index: int = -100,
                 augmentation_prob: float = 0.01, alphabet: List[str] = None):
        """
        Args:
            sequences: List of DNA sequences
            tokenizer: Tokenizer with encode method (LegacyTokenizer)
            max_length: Maximum sequence length
            masking_prob: Probability of masking a token
            replace_prob: Probability of replacing masked token with [MASK]
            random_prob: Probability of replacing masked token with random token
            ignore_index: Index for tokens to ignore in loss calculation
            augmentation_prob: Probability of sequence augmentation (same as finetuning)
            alphabet: DNA alphabet for augmentation
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking_prob = masking_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.ignore_index = ignore_index
        
        # Augmentation settings
        self.augmentation_prob = augmentation_prob
        self.alphabet = alphabet if alphabet is not None else ['A', 'C', 'G', 'T']
        self.use_augmentation = AUGMENTATION_AVAILABLE and augmentation_prob > 0
        
        # Get special token IDs
        self.pad_id = tokenizer.vocab.get('[PAD]', 0)
        self.mask_id = tokenizer.vocab.get('[MASK]', 4)
        self.vocab_size = len(tokenizer.vocab)
        
        logger.info(f"MLM Dataset: {len(sequences)} sequences, masking_prob={masking_prob}")
        logger.info(f"Augmentation: enabled={self.use_augmentation}, prob={augmentation_prob}")
        logger.info(f"Special tokens: PAD={self.pad_id}, MASK={self.mask_id}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Apply sequence augmentation (consistent with finetuning)
        if self.use_augmentation:
            sequence = augment_sequence(
                seq=sequence, 
                alphabet=self.alphabet, 
                modification_probability=self.augmentation_prob
            )
        
        # Tokenize sequence
        encoding = self.tokenizer.encode(
            sequence, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Create MLM labels and input
        input_ids, labels = self._apply_mlm_masking(input_ids, attention_mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _apply_mlm_masking(self, input_ids: List[int], attention_mask: List[int]):
        """
        Apply MLM masking strategy:
        - 80% of time: replace with [MASK]
        - 10% of time: replace with random token
        - 10% of time: keep original token
        """
        input_ids = input_ids.copy()
        labels = [self.ignore_index] * len(input_ids)
        
        for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
            # Skip padding tokens
            if mask == 0 or token_id == self.pad_id:
                continue
            
            # Skip special tokens (CLS, SEP, etc.)
            if token_id in [self.tokenizer.vocab.get('[CLS]', 2), 
                           self.tokenizer.vocab.get('[SEP]', 3)]:
                continue
            
            # Decide whether to mask this token
            if random.random() < self.masking_prob:
                labels[i] = token_id  # Store original token in labels
                
                rand = random.random()
                if rand < self.replace_prob:
                    # 80% chance: replace with [MASK]
                    input_ids[i] = self.mask_id
                elif rand < self.replace_prob + self.random_prob:
                    # 10% chance: replace with random token
                    # Get random token (avoid special tokens)
                    random_token = random.randint(5, self.vocab_size - 1)
                    input_ids[i] = random_token
                # else: 10% chance: keep original token
        
        return input_ids, labels


def load_pretraining_data(data_path: Path, dataset_size: str) -> List[str]:
    """
    Load sequences for pretraining (no labels needed)
    
    Args:
        data_path: Path to CSV file with sequences
        dataset_size: "debug_5genera_10fold" or "full_10fold"
        
    Returns:
        List of DNA sequences
    """
    logger.info(f"Loading pretraining data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} sequences from {data_path}")
    
    # Extract just the sequences (no labels needed for pretraining)
    sequences = df['sequence'].tolist()
    
    # Remove any NaN sequences
    sequences = [seq for seq in sequences if pd.notna(seq) and len(seq) > 0]
    
    logger.info(f"Final pretraining dataset: {len(sequences)} sequences")
    return sequences


def create_mlm_data_loaders(data_path: Path, tokenizer, config: Dict) -> tuple:
    """
    Create train and validation data loaders for MLM pretraining
    
    Args:
        data_path: Path to CSV data file  
        tokenizer: Tokenizer instance
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    # Load all sequences
    sequences = load_pretraining_data(data_path, config['experiment']['dataset_size'])
    
    # Split into train/val (90%/10% for pretraining)
    random.seed(config.get('seed', 42))
    random.shuffle(sequences)
    
    split_idx = int(0.9 * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    logger.info(f"Pretraining split: {len(train_sequences)} train, {len(val_sequences)} val")
    
    # Create datasets
    augmentation_prob = config['preprocessing'].get('base_modification_probability', 0.01)
    alphabet = config['preprocessing'].get('alphabet', ['A', 'C', 'G', 'T'])
    
    train_dataset = MLMDataset(
        train_sequences, 
        tokenizer,
        max_length=config['preprocessing']['max_length'],
        masking_prob=config['preprocessing'].get('masking_percentage', 0.15),
        augmentation_prob=augmentation_prob,
        alphabet=alphabet
    )
    
    val_dataset = MLMDataset(
        val_sequences,
        tokenizer, 
        max_length=config['preprocessing']['max_length'],
        masking_prob=config['preprocessing'].get('masking_percentage', 0.15),
        augmentation_prob=augmentation_prob,
        alphabet=alphabet
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader