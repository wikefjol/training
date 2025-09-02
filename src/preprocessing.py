"""
Preprocessing utilities for DNA sequences
"""

import re
from typing import Dict, List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class KmerTokenizer:
    """K-mer based tokenizer for DNA sequences"""
    
    def __init__(self, k: int = 6, stride: int = 1, vocab_size: int = 1024):
        """
        Args:
            k: K-mer size
            stride: Stride for k-mer extraction
            vocab_size: Maximum vocabulary size
        """
        self.k = k
        self.stride = stride
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [
            self.pad_token,
            self.unk_token, 
            self.cls_token,
            self.sep_token,
            self.mask_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
    
    def build_vocab(self, sequences: List[str]):
        """
        Build vocabulary from sequences
        
        Args:
            sequences: List of DNA sequences
        """
        logger.info(f"Building vocabulary from {len(sequences)} sequences")
        
        # Count k-mers
        kmer_counts = Counter()
        for seq in sequences:
            seq = seq.upper()
            for i in range(0, len(seq) - self.k + 1, self.stride):
                kmer = seq[i:i + self.k]
                if re.match(r'^[ACGT]+$', kmer):  # Valid DNA k-mer
                    kmer_counts[kmer] += 1
        
        # Select most common k-mers
        vocab_size = min(self.vocab_size - len(self.vocab), len(kmer_counts))
        most_common = kmer_counts.most_common(vocab_size)
        
        # Add to vocabulary
        current_idx = len(self.vocab)
        for kmer, _ in most_common:
            if kmer not in self.vocab:
                self.vocab[kmer] = current_idx
                self.reverse_vocab[current_idx] = kmer
                current_idx += 1
        
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens")
    
    def encode(self, sequence: str, max_length: int = 512, 
               padding: str = 'max_length', truncation: bool = True) -> Dict:
        """
        Encode a DNA sequence
        
        Args:
            sequence: DNA sequence
            max_length: Maximum length
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        sequence = sequence.upper()
        
        # Extract k-mers
        tokens = [self.cls_token]
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i + self.k]
            if re.match(r'^[ACGT]+$', kmer):
                tokens.append(kmer)
        tokens.append(self.sep_token)
        
        # Truncate if needed
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token]
        
        # Convert to ids
        input_ids = []
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab[self.unk_token])
        
        # Pad if needed
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length':
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.vocab[self.pad_token]] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to sequence"""
        tokens = []
        for idx in token_ids:
            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                if token not in [self.pad_token, self.cls_token, 
                               self.sep_token, self.unk_token]:
                    tokens.append(token)
        
        # Reconstruct sequence (approximate due to overlapping k-mers)
        if not tokens:
            return ""
        
        sequence = tokens[0]
        for token in tokens[1:]:
            # Try to find overlap
            for overlap in range(self.k - 1, 0, -1):
                if sequence[-overlap:] == token[:overlap]:
                    sequence += token[overlap:]
                    break
            else:
                sequence += token
        
        return sequence


class CharacterTokenizer:
    """Simple character-level tokenizer for DNA sequences"""
    
    def __init__(self):
        """Initialize character tokenizer"""
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4,
            'A': 5,
            'C': 6,
            'G': 7,
            'T': 8,
            'N': 9
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
    
    def build_vocab(self, sequences: List[str]):
        """No-op for character tokenizer (vocab is fixed)"""
        pass
    
    def encode(self, sequence: str, max_length: int = 512,
               padding: str = 'max_length', truncation: bool = True) -> Dict:
        """
        Encode a DNA sequence at character level
        
        Args:
            sequence: DNA sequence
            max_length: Maximum length
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        sequence = sequence.upper()
        
        # Convert to tokens
        tokens = [self.cls_token]
        for char in sequence:
            if char in 'ACGTN':
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
        tokens.append(self.sep_token)
        
        # Truncate if needed
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token]
        
        # Convert to ids
        input_ids = [self.vocab.get(token, self.vocab[self.unk_token]) 
                    for token in tokens]
        
        # Pad if needed
        attention_mask = [1] * len(input_ids)
        if padding == 'max_length':
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.vocab[self.pad_token]] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def create_tokenizer(config: Dict, sequences: Optional[List[str]] = None):
    """
    Create tokenizer based on config
    
    Args:
        config: Configuration dictionary
        sequences: Optional sequences for building vocabulary
        
    Returns:
        Tokenizer instance
    """
    tokenizer_type = config['preprocessing']['tokenizer']
    
    if tokenizer_type == 'kmer':
        tokenizer = KmerTokenizer(
            k=config['preprocessing']['kmer_size'],
            stride=config['preprocessing']['stride'],
            vocab_size=config['model']['vocab_size']
        )
        if sequences:
            tokenizer.build_vocab(sequences)
    elif tokenizer_type == 'character':
        tokenizer = CharacterTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    return tokenizer