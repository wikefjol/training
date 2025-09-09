#!/usr/bin/env python3
"""
Test script to verify legacy preprocessing works correctly
Tests with alphabet ABC...Z to ensure pipeline functions properly
"""

import sys
import os
import random
import logging
from typing import Protocol, List, Any

# Import legacy preprocessing directly without src module dependencies
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import legacy_preprocessing

# Simple vocabulary class for testing
class SimpleVocabulary:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0
        
        # Add PAD token
        self._add_token('PAD')
    
    def _add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.token_to_id[token]
    
    def map_sentence(self, tokenized_sentence):
        """Map tokenized sentence to IDs"""
        result = []
        for token_list in tokenized_sentence:
            token = token_list[0] if token_list else 'PAD'  # Extract token from list
            token_id = self._add_token(token)  # Auto-add if not seen
            result.append([token_id])
        return result
    
    def __len__(self):
        return len(self.token_to_id)

def test_legacy_preprocessing():
    """Test legacy preprocessing with alphabet sequence"""
    
    print("🧪 Testing Legacy Preprocessing Pipeline")
    print("=" * 50)
    
    # Test sequence: alphabet
    test_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"Input sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)} characters")
    
    # Configuration matching your old setup
    k = 3  # 3-mer tokenization
    optimal_length = 10  # Small for testing - equivalent to ~30 bases
    modification_probability = 0.01  # Small chance for testing
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    print(f"\\nConfiguration:")
    print(f"- K-mer size: {k}")
    print(f"- Optimal length: {optimal_length} tokens")
    print(f"- Modification probability: {modification_probability}")
    print(f"- Alphabet: {len(alphabet)} characters")
    
    # Create vocabulary
    print("\\n📚 Creating vocabulary...")
    vocab = SimpleVocabulary()
    print(f"✅ Vocabulary created with {len(vocab)} tokens")
    
    # Test both training and validation preprocessors
    for training_mode in [True, False]:
        mode_name = "Training" if training_mode else "Validation"
        print(f"\\n🔧 Testing {mode_name} Mode:")
        
        preprocessor = legacy_preprocessing.create_legacy_preprocessor(
            k=k,
            optimal_length=optimal_length,
            alphabet=alphabet,
            vocab=vocab
        )
        
        # Process the sequence
        try:
            result = preprocessor.process(test_sequence)
            print(f"✅ Processing successful!")
            print(f"   Input length: {len(test_sequence)} chars")
            print(f"   Output length: {len(result)} tokens")
            print(f"   Output shape: {len(result)}x{len(result[0]) if result else 0}")
            print(f"   First 5 tokens: {result[:5] if len(result) > 5 else result}")
            
            # Verify output properties
            assert len(result) == optimal_length, f"Expected {optimal_length} tokens, got {len(result)}"
            assert all(len(token) == 1 for token in result), "Each token should be a single integer"
            print(f"✅ All assertions passed!")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
    
    print("\\n🎉 All tests passed! Legacy preprocessing is working correctly.")
    
    # Additional test: show the actual tokenization process
    print("\\n🔍 Detailed Pipeline Breakdown:")
    print("-" * 30)
    
    # Create a simple preprocessor for step-by-step analysis
    preprocessor = legacy_preprocessing.create_legacy_preprocessor(k=k, optimal_length=optimal_length,
                                            alphabet=alphabet, vocab=vocab)
    
    # Step by step processing
    sequence_list = list(test_sequence)
    print(f"1. String to list: {sequence_list[:10]}... (length: {len(sequence_list)})")
    
    # No augmentation step (removed)
    print(f"2. Augmentation: SKIPPED (handled at dataset level)")
    
    # Tokenization
    tokenized = preprocessor.tokenization_strategy.execute(sequence_list)
    print(f"3. After tokenization: {tokenized[:5]}... (length: {len(tokenized)})")
    
    # Padding
    padded = preprocessor.padding_strategy.execute(tokenized)
    print(f"4. After padding: {padded[:5]}... (length: {len(padded)})")
    
    # Truncation
    truncated = preprocessor.truncation_strategy.execute(padded)
    print(f"5. After truncation: {truncated[:5]}... (length: {len(truncated)})")
    
    # Vocabulary mapping
    mapped = vocab.map_sentence(truncated)
    print(f"6. After vocab mapping: {mapped[:5]}... (length: {len(mapped)})")
    
    return True

if __name__ == "__main__":
    success = test_legacy_preprocessing()
    sys.exit(0 if success else 1)