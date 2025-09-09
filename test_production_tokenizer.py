#!/usr/bin/env python3
"""
Test LegacyTokenizer with production settings (ACGT alphabet, optimal_length=500)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import legacy_preprocessing

def test_production_tokenizer():
    print("🧪 Testing Production LegacyTokenizer")
    print("=" * 50)
    
    # Test sequence (DNA)
    test_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"  # 42 bases
    print(f"Input sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)} bases")
    
    # Production settings
    tokenizer = legacy_preprocessing.LegacyTokenizer(
        k=3,
        optimal_length=500,  # 1500 bases / 3 = 500 tokens
        modification_probability=0.01,  # Used in dataset-level augmentation
        alphabet=['A', 'C', 'G', 'T']   # Nucleotide alphabet
    )
    
    print(f"\nProduction Tokenizer:")
    print(f"- K-mer size: {tokenizer.k}")
    print(f"- Optimal length: {tokenizer.optimal_length} tokens")
    print(f"- Vocabulary size: {len(tokenizer.vocab)} (should be 64 k-mers + 5 special = 69)")
    print(f"- Alphabet: {tokenizer.alphabet}")
    
    # Expected vocab: 4^3 = 64 k-mers + 5 special tokens = 69 total
    expected_vocab_size = 4**3 + 5
    assert len(tokenizer.vocab) == expected_vocab_size, f"Expected {expected_vocab_size}, got {len(tokenizer.vocab)}"
    
    # Test encoding
    encoding = tokenizer.encode(test_sequence, max_length=512, padding='max_length', truncation=True)
    
    print(f"\nEncoding results:")
    print(f"- Input IDs length: {len(encoding['input_ids'])}")
    print(f"- Attention mask length: {len(encoding['attention_mask'])}")
    print(f"- Expected length (optimal_length): {tokenizer.optimal_length}")
    print(f"- Match: {len(encoding['input_ids']) == tokenizer.optimal_length}")
    
    # Show first few tokens
    print(f"- First 10 input IDs: {encoding['input_ids'][:10]}")
    print(f"- Last 10 input IDs: {encoding['input_ids'][-10:]}")
    print(f"- All attention mask = 1: {all(x == 1 for x in encoding['attention_mask'])}")
    
    # Test that we get consistent k-mer tokenization
    expected_kmers = 42 // 3  # 14 k-mers from 42-base sequence
    print(f"\nTokenization verification:")
    print(f"- Input: {len(test_sequence)} bases")
    print(f"- Expected k-mers: ~{expected_kmers}")
    print(f"- Padded to: {tokenizer.optimal_length} tokens")
    
    return True

if __name__ == "__main__":
    success = test_production_tokenizer()
    sys.exit(0 if success else 1)