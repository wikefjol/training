#!/usr/bin/env python3
"""
Pretrain a BERT-like model using Masked Language Modeling (MLM)
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.mlm_data import create_mlm_data_loaders
from src.legacy_preprocessing import LegacyTokenizer
from src.model import create_model
from src.mlm_trainer import MLMTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pretraining.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config():
    """Load configuration from .env and config.yaml"""
    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent / '.env.example'
    load_dotenv(env_path)
    
    # Get paths from environment
    paths = {
        'data_root': os.path.expandvars(os.getenv('DATA_ROOT')),
        'experiments_dir': os.path.expandvars(os.getenv('EXPERIMENTS_DIR')),
        'models_dir': os.path.expandvars(os.getenv('MODELS_DIR')),
        'logs_dir': os.path.expandvars(os.getenv('LOGS_DIR'))
    }
    
    return paths


def pretrain_model(config_path: str, paths: dict):
    """
    Pretrain a model using MLM
    
    Args:
        config_path: Path to pretraining config file
        paths: Paths dictionary from environment
    """
    # Load config
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Pretraining configuration: {config['experiment']}")
    
    # Build data path based on experiment structure
    experiment_base = Path(paths['experiments_dir']) / config['experiment']['fold_type'] / \
                      config['experiment']['dataset_size']
    
    data_path = experiment_base / 'data' / f"{config['experiment']['union_type']}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading pretraining data from {data_path}")
    
    # Create tokenizer
    logger.info("Building legacy tokenizer...")
    optimal_length = config['preprocessing']['max_length'] // config['preprocessing']['kmer_size']
    
    tokenizer = LegacyTokenizer(
        k=config['preprocessing']['kmer_size'],
        optimal_length=optimal_length,
        modification_probability=config['preprocessing'].get('base_modification_probability', 0.01),
        alphabet=config['preprocessing']['alphabet']
    )
    
    # Update vocab size in config
    config['model']['vocab_size'] = len(tokenizer.vocab)
    logger.info(f"Using legacy preprocessing with exhaustive k-mer vocabulary")
    logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
    logger.info(f"Optimal sequence length: {tokenizer.optimal_length} tokens")
    
    # Create data loaders
    logger.info("Creating MLM data loaders...")
    train_loader, val_loader = create_mlm_data_loaders(data_path, tokenizer, config)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model (no num_classes needed for pretraining)
    logger.info("Creating model for pretraining...")
    model = create_model(
        vocab_size=config['model']['vocab_size'],
        num_classes=None,  # Not needed for pretraining
        config=config
    )
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters "
                f"({total_params:,} total)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_size = config['experiment']['dataset_size']
    output_dir = Path(paths['models_dir']) / 'pretraining' / f"{dataset_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    # Create trainer
    logger.info("Creating MLM trainer...")
    trainer = MLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=str(output_dir),
        accumulation_steps=config['training'].get('accumulation_steps', 1),
        debug=config.get('debug', False)
    )
    
    # Train
    logger.info("Starting pretraining...")
    start_time = datetime.now()
    results = trainer.train()
    end_time = datetime.now()
    
    # Save results
    results['training_time'] = str(end_time - start_time)
    results['timestamp'] = datetime.now().isoformat()
    
    logger.info("Pretraining complete!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Training time: {results['training_time']}")
    logger.info(f"Model saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Pretrain a model with MLM")
    parser.add_argument('--config', type=str, required=True, help='Path to pretraining config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load paths
    paths = load_config()
    
    # Pretrain model
    results = pretrain_model(args.config, paths)
    
    # Print summary
    print("\n" + "="*60)
    print("Pretraining Complete")
    print("="*60)
    print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    print(f"Final Training Loss: {results['final_train_loss']:.4f}")
    print(f"Final Validation Loss: {results['final_val_loss']:.4f}")
    print(f"Epochs Completed: {results['epochs_completed']}")
    print(f"Training Time: {results['training_time']}")
    print("="*60)


if __name__ == "__main__":
    main()