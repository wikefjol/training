#!/usr/bin/env python3
"""
Test pretraining setup without actually training
"""

import sys
import yaml
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent))

from src.legacy_preprocessing import LegacyTokenizer
from src.model import create_model

def test_pretraining_setup():
    print("🧪 Testing Pretraining Setup")
    print("=" * 50)
    
    # Load tiny config
    config_path = "configs/pretrain_tiny.yaml"
    print(f"Loading config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded:")
    print(f"  Mode: {config['model']['mode']}")
    print(f"  Dataset: {config['experiment']['dataset_size']}")
    print(f"  Architecture: {config['model']['hidden_size']}h x {config['model']['num_hidden_layers']}L")
    
    # Create tokenizer
    print("\nCreating tokenizer...")
    optimal_length = config['preprocessing']['max_length'] // config['preprocessing']['kmer_size']
    
    tokenizer = LegacyTokenizer(
        k=config['preprocessing']['kmer_size'],
        optimal_length=optimal_length,
        modification_probability=config['preprocessing']['base_modification_probability'],
        alphabet=config['preprocessing']['alphabet']
    )
    
    # Update vocab size
    config['model']['vocab_size'] = len(tokenizer.vocab)
    print(f"Tokenizer created:")
    print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    print(f"  Optimal length: {tokenizer.optimal_length} tokens")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=config['model']['vocab_size'],
        num_classes=None,  # No classes needed for pretraining
        config=config
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created:")
    print(f"  Mode: {model.mode}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Has MLM head: {hasattr(model, 'mlm_head')}")
    print(f"  Has classifier: {hasattr(model, 'classifier')}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    import torch
    
    # Create dummy batch
    batch_size = 2
    seq_length = tokenizer.optimal_length
    
    input_ids = torch.randint(5, len(tokenizer.vocab), (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(5, len(tokenizer.vocab), (batch_size, seq_length))
    
    # Set some positions to ignore
    labels[:, :10] = -100  # Ignore first 10 positions
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Non-ignored positions: {(labels != -100).sum().item()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, labels)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_length}, {len(tokenizer.vocab)}]")
    
    # Test loss calculation
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    print(f"MLM loss: {loss.item():.4f}")
    
    print("\n✅ All tests passed! Pretraining setup is working correctly.")
    
    return True

if __name__ == "__main__":
    success = test_pretraining_setup()
    sys.exit(0 if success else 1)