"""
Preprocessing utilities for DNA sequences
"""

import re
from typing import Dict, List, Optional
from collections import Counter
from itertools import product
import logging

logger = logging.getLogger(__name__)


class KmerTokenizer:
    """K-mer based tokenizer for DNA sequences with exhaustive alphabet vocabulary"""
    
    def __init__(self, k: int = 3, stride: int = 3, alphabet: List[str] = None):
        """
        Args:
            k: K-mer size
            stride: Stride for k-mer extraction  
            alphabet: DNA alphabet (default: ['A', 'C', 'G', 'T'])
        """
        self.k = k
        self.stride = stride
        self.alphabet = alphabet if alphabet is not None else ['A', 'C', 'G', 'T']
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        
        # Initialize with special tokens and exhaustive k-mer vocabulary
        self._init_special_tokens()
        self._build_exhaustive_vocab()
    
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
    
    def _build_exhaustive_vocab(self):
        """Build exhaustive vocabulary from all possible k-mers in alphabet"""
        logger.info(f"Building exhaustive {self.k}-mer vocabulary from alphabet {self.alphabet}")
        
        # Generate all possible k-mers
        current_idx = len(self.vocab)  # Start after special tokens
        
        for kmer_tuple in product(self.alphabet, repeat=self.k):
            kmer = ''.join(kmer_tuple)
            self.vocab[kmer] = current_idx
            self.reverse_vocab[current_idx] = kmer
            current_idx += 1
        
        logger.info(f"Exhaustive vocabulary built with {len(self.vocab)} tokens ({len(self.vocab) - 5} k-mers + 5 special tokens)")
    
    def build_vocab(self, sequences: List[str]):
        """
        No-op for exhaustive tokenizer (vocab is already built from alphabet)
        
        Args:
            sequences: List of DNA sequences (ignored)
        """
        logger.info("Using exhaustive vocabulary - no sequence-based vocab building needed")
    
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
    
import random
from typing import List, Optional, Sequence

def augment_sequence(seq: str,
                          alphabet: Sequence[str],
                          modification_probability: float = 0.05,
                          weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
                          rng: Optional[random.Random] = None) -> str:
    """
    Apply random augmentations (insert, replace, delete, swap) to a DNA/protein
    sequence with fixed probabilities.

    Args:
        seq (str): Input sequence.
        alphabet (Sequence[str]): Valid symbols for insert/replace operations.
        modification_probability (float): Per-position chance of applying an operation.
        weights (Sequence[float]): Relative probabilities for (insert, replace, delete, swap).
        rng (Optional[random.Random]): Optional RNG instance for reproducibility.

    Returns:
        str: Augmented sequence.

    Notes:
        - Each position is independently considered for modification.
        - Insert may occur at the current or next index.
        - Delete is skipped if sequence length would drop below 1.
        - Swap exchanges the symbol at the current index with a neighbor.
    """

    r = rng or random
    s = list(seq)
    ops = ("insert", "replace", "delete", "swap")
    i = 0
    while i < len(s):
        if r.random() < modification_probability:
            op = r.choices(ops, weights=weights, k=1)[0]
            if op == "insert":
                ins_idx = r.choice((i, i + 1))
                if ins_idx <= len(s):
                    s.insert(ins_idx, r.choice(alphabet))
            elif op == "replace":
                s[i] = r.choice(alphabet)
            elif op == "delete":
                if len(s) > 1:
                    s.pop(i)
                    i += 1  # advance to avoid re-hitting same position repeatedly
                    continue
            elif op == "swap":
                j = i + r.choice((-1, 1))
                if 0 <= j < len(s):
                    s[i], s[j] = s[j], s[i]
        i += 1
    return "".join(s)
