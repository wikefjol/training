#!/usr/bin/env python3
"""
Test the LegacyTokenizer wrapper
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import legacy_preprocessing

def test_legacy_tokenizer():
    print("🧪 Testing LegacyTokenizer Wrapper")
    print("=" * 50)
    
    # Test sequence
    test_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"Input sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)} characters")
    
    # Create tokenizer with same config as full models
    tokenizer = legacy_preprocessing.LegacyTokenizer(
        k=3,
        optimal_length=167,  # 500 tokens ÷ 3 ≈ 167
        modification_probability=0.01,
        alphabet=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    )
    
    print(f"\nTokenizer created:")
    print(f"- K-mer size: {tokenizer.k}")
    print(f"- Optimal length: {tokenizer.optimal_length}")
    print(f"- Vocabulary size: {len(tokenizer.vocab)}")
    print(f"- Alphabet size: {len(tokenizer.alphabet)}")
    
    # Test encoding
    encoding = tokenizer.encode(test_sequence, max_length=512, padding='max_length', truncation=True)
    
    print(f"\nEncoding results:")
    print(f"- Input IDs length: {len(encoding['input_ids'])}")
    print(f"- Attention mask length: {len(encoding['attention_mask'])}")
    print(f"- First 10 input IDs: {encoding['input_ids'][:10]}")
    print(f"- First 10 attention mask: {encoding['attention_mask'][:10]}")
    print(f"- All attention mask values are 1: {all(x == 1 for x in encoding['attention_mask'])}")
    
    # Verify consistency with legacy preprocessing
    print(f"\nVerification:")
    print(f"- Expected length (optimal_length): {tokenizer.optimal_length}")
    print(f"- Actual length: {len(encoding['input_ids'])}")
    print(f"- Match: {len(encoding['input_ids']) == tokenizer.optimal_length}")
    
    return True

if __name__ == "__main__":
    success = test_legacy_tokenizer()
    sys.exit(0 if success else 1)