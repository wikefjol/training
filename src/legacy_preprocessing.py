"""
Legacy preprocessing implementation - exact copy from bin/preprocessing/
Implements: BaseStrategy augmentation, KmerStrategy tokenization, 
SlidingwindowStrategy truncation, EndStrategy padding
"""

import random
import logging
from typing import Protocol, List, Any, Dict

# Logging setup (simplified from original)
system_logger = logging.getLogger("system_logger")

def with_logging(level=8):
    """Simplified logging decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =================== AUGMENTATION ===================

class SequenceModifier():
    """ Modifies a sequence at a specific position"""
    def __init__(self, alphabet: list[str]):
        self.alphabet = alphabet
    
    @with_logging(level=8)
    def _insert(self, seq: list[str], idx: int) -> None:
        insert_idx = random.choice([idx, idx + 1])
        if insert_idx <= len(seq):
            seq.insert(insert_idx, random.choice(self.alphabet))
    
    @with_logging(level=8)
    def _replace(self, seq: list[str], idx: int) -> None:
        seq[idx] = random.choice(self.alphabet)
    
    @with_logging(level=8)
    def _delete(self, seq: list[str], idx: int) -> None:
        if len(seq) > 1:
            seq.pop(idx)
    
    @with_logging(level=8)
    def _swap(self, seq: list[str], idx: int) -> None:
        swap_pos = idx + random.choice([-1, 1])
        if 0 <= swap_pos < len(seq):
            seq[idx], seq[swap_pos] = seq[swap_pos], seq[idx]

class BaseStrategy():
    """ Standard augmentation strategy"""
    def __init__(self, modifier: SequenceModifier, alphabet: list[str], modification_probability: float = 0.05):
        self.alphabet = alphabet
        self.modifier = modifier
        self.modification_probability = modification_probability
        self.operation_map = {
            'insert': self.modifier._insert,
            'replace': self.modifier._replace,
            'delete': self.modifier._delete,
            'swap': self.modifier._swap
        }
        self.operations = list(self.operation_map.keys())
        self.weights = [0.25, 0.25, 0.25, 0.25]
    
    @with_logging(level=9)
    def execute(self, seq: list[str]) -> list[str]:
        augmented_seq = seq[:]
        for pos, _ in enumerate(augmented_seq):
            
            should_modify = random.random() < self.modification_probability
            if should_modify:     
                operation = random.choices(self.operations, weights=self.weights, k=1)[0]
                self.operation_map[operation](augmented_seq, pos)
        return augmented_seq

class IdentityStrategy(BaseStrategy):
    """ Do-nothing augmentation"""
    @with_logging(level=9)
    def execute(self, seq: list[str]) -> list[str]:
        return seq[:]

# =================== TOKENIZATION ===================

class KmerStrategy:
    def __init__(self, k: int, padding_alphabet: list[str] = ['A','C','G','T']):
        self.k = k
        self.padding_alphabet = padding_alphabet
    
    @with_logging(level=8)
    def _make_divisible_by_k(self, input_seq: list[str]) -> list[str]:
        """Pads seq if its length is not divisible by k, using random characters from the padding alphabet."""
        seq = input_seq[:]
        while len(seq) % self.k != 0:
           seq.append(random.choice(self.padding_alphabet))
        return seq
    
    @with_logging(level=9)
    def execute(self, input_seq: list[str]) -> list[list[str]]:
        """Tokenizes the seq into k-mers, adding padding if necessary."""

        seq = input_seq[:]
        if self.k == 0:
            self.k = 1
            system_logger.warning(f"[PROCESSING.TOKENIZATION] 'KmerStrategy.execute':  k cant be {0}; set to {1}")

        remainder = len(seq) % self.k
        if remainder != 0:
            seq = self._make_divisible_by_k(seq)

        # Tokenize into k-mers
        kmer_seq = [''.join(seq[i:i + self.k]) for i in range(0, len(seq), self.k)]
        tokenized_sentence = [[kmer] for kmer in kmer_seq]

        return tokenized_sentence

# =================== TRUNCATION ===================

class SlidingwindowStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        truncated_seq = seq[:]

        # Return the whole sequence if it's already within optimal length
        if len(truncated_seq) <= self.optimal_length:
            return truncated_seq  
    
        max_start_index = len(truncated_seq) - self.optimal_length
        
        # Choose a random start index in the valid range
        start_index = random.randint(0, max_start_index)
        
        # Extract the subarray of length optimal_length
        return truncated_seq[start_index:start_index + self.optimal_length]

class EndTruncationStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        return seq[:self.optimal_length]

class FrontTruncationStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        return seq[-self.optimal_length:]

# =================== PADDING ===================

class EndPaddingStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        # Create a copy to avoid modifying the original sequence
        padded_seq = seq[:]
        
        # Add ['PAD'] sub-lists until the length matches optimal_length
        while len(padded_seq) < self.optimal_length:
            padded_seq.append(['PAD'])
        
        # If seq is already longer than optimal_length, trim it
        return padded_seq

# =================== PREPROCESSOR ===================

class Strategy(Protocol):
    """Augments sequence by imitating sequencing errors"""
    def execute(self, sequence: list[str]) -> list[str]:
        """
        Parameters
        ----------
        sequence : str
            DNA sequence

        Returns
        ----------
        sequence : str
            Augmented DNA sequence
        """

class LegacyPreprocessor:
    def __init__(
        self,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        vocab = None
    ):
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab
        print("DEBUG: Using legacy preprocessor (no augmentation)")

    def process(self, sequence: str) -> List[List[str]]:
        sequence = list(sequence)  # Convert string to list of characters
        # Skip augmentation - handled at dataset level
        tokenized_sentence: List[List[str]] = self.tokenization_strategy.execute(sequence)
        padded_sentence: List[List[str]] = self.padding_strategy.execute(tokenized_sentence)
        processed_sentence: List[List[str]] = self.truncation_strategy.execute(padded_sentence)
        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)
        
        return mapped_sentence

# =================== TOKENIZER WRAPPER ===================

class LegacyTokenizer:
    """Wrapper to make legacy preprocessor compatible with current tokenizer interface"""
    
    def __init__(self, k: int = 3, optimal_length: int = 167, 
                 modification_probability: float = 0.01, alphabet: List[str] = None):
        self.k = k
        self.optimal_length = optimal_length 
        self.modification_probability = modification_probability
        self.alphabet = alphabet if alphabet is not None else ['A', 'C', 'G', 'T']
        
        # Build exhaustive k-mer vocabulary like KmerTokenizer
        from itertools import product
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        
        # Initialize vocab
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        # Add exhaustive k-mers
        current_idx = len(self.vocab)
        for kmer_tuple in product(self.alphabet, repeat=self.k):
            kmer = ''.join(kmer_tuple)
            self.vocab[kmer] = current_idx
            self.reverse_vocab[current_idx] = kmer
            current_idx += 1
        
        # Create simple vocab object for legacy preprocessor
        class SimpleVocab:
            def __init__(self, vocab_dict):
                self.vocab_dict = vocab_dict
                
            def map_sentence(self, tokenized_sentence):
                result = []
                for token_list in tokenized_sentence:
                    token = token_list[0] if token_list else 'PAD'
                    if token in self.vocab_dict:
                        result.append([self.vocab_dict[token]])
                    else:
                        result.append([self.vocab_dict['[UNK]']])
                return result
        
        self.simple_vocab = SimpleVocab(self.vocab)
        
        # Create single preprocessor (no augmentation - handled at dataset level)
        self.preprocessor = create_legacy_preprocessor(
            k=k, optimal_length=optimal_length,
            alphabet=alphabet, vocab=self.simple_vocab
        )
    
    def encode(self, sequence: str, max_length: int = 512,
               padding: str = 'max_length', truncation: bool = True) -> Dict:
        """
        Encode using legacy preprocessing but return current format
        
        Args:
            sequence: DNA sequence
            max_length: Maximum length (ignored - uses optimal_length)
            padding: Padding strategy (ignored - legacy handles this)
            truncation: Whether to truncate (ignored - legacy handles this)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Use single preprocessor (no augmentation - handled at dataset level)
        processed = self.preprocessor.process(sequence)
        
        # Flatten the nested list structure
        input_ids = [token[0] for token in processed]
        
        # Create attention mask (all 1s since legacy preprocessor handles padding)
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# =================== FACTORY FUNCTIONS ===================

def create_legacy_preprocessor(k: int, optimal_length: int, alphabet: List[str], vocab):
    """
    Create a legacy preprocessor without augmentation:
    - KmerStrategy tokenization (non-overlapping)
    - SlidingwindowStrategy truncation (random window)
    - EndPaddingStrategy padding
    
    Note: Augmentation is handled at dataset level using augment_sequence()
    """
    
    # Strategies (no augmentation)
    tokenization_strategy = KmerStrategy(k, alphabet)
    truncation_strategy = SlidingwindowStrategy(optimal_length)
    padding_strategy = EndPaddingStrategy(optimal_length)
    
    return LegacyPreprocessor(
        tokenization_strategy=tokenization_strategy,
        padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy,
        vocab=vocab
    )