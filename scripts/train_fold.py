#!/usr/bin/env python3
"""
Train a single fold of the k-fold cross-validation
"""

import os
import sys
import json
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
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_fold_data, prepare_data_for_training, create_data_loaders
from src.preprocessing import KmerTokenizer
from src.model import create_model
from src.trainer import Trainer
from src.hierarchical_trainer import HierarchicalTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
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
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env.example'
    load_dotenv(env_path)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from environment
    paths = {
        'data_root': os.path.expandvars(os.getenv('DATA_ROOT')),
        'experiments_dir': os.path.expandvars(os.getenv('EXPERIMENTS_DIR')),
        'models_dir': os.path.expandvars(os.getenv('MODELS_DIR')),
        'logs_dir': os.path.expandvars(os.getenv('LOGS_DIR'))
    }
    
    return config, paths


def train_fold(fold: int, config: dict, paths: dict):
    """
    Train a single fold
    
    Args:
        fold: Fold number (1-10)
        config: Configuration dictionary
        paths: Paths dictionary
    """
    logger.info(f"Starting training for fold {fold}")
    logger.info(f"Configuration: {json.dumps(config['experiment'], indent=2)}")
    
    # Build experiment base path
    experiment_base = Path(paths['experiments_dir']) / config['experiment']['fold_type'] / \
                      config['experiment']['dataset_size']
    
    # Data is now in data/ subdirectory
    data_path = experiment_base / 'data' / f"{config['experiment']['union_type']}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    
    # Load fold data
    train_df, val_df = load_fold_data(
        data_path, fold, config['experiment']['fold_type']
    )
    
    # Load global label encoders
    encoder_path = data_path.parent / f"label_encoders_{config['experiment']['union_type']}_global.json"
    label_encoders = None
    encoder_dicts = None
    if encoder_path.exists():
        logger.info(f"Loading label encoders from {encoder_path}")
        from src.data import LabelEncoder
        
        with open(encoder_path, 'r') as f:
            encoder_dicts = json.load(f)
        
        # Log unknown label statistics
        for level, enc_dict in encoder_dicts.items():
            if enc_dict.get('num_unknown_in_val', 0) > 0:
                logger.warning(f"  {level}: {enc_dict['num_unknown_in_val']} unknown labels in validation")
                if 'example_unknown' in enc_dict:
                    logger.debug(f"    Examples: {enc_dict['example_unknown'][:3]}")
    else:
        logger.warning(f"Label encoder file not found: {encoder_path}")
        logger.warning("Will build encoders from training data (old behavior)")
    
    # Prepare data
    train_data = prepare_data_for_training(train_df)
    val_data = prepare_data_for_training(val_df)
    
    logger.info(f"Train: {train_data['num_sequences']} sequences, {train_data['num_classes']} classes")
    logger.info(f"Val: {val_data['num_sequences']} sequences, {val_data['num_classes']} classes")
    
    # Create tokenizer
    logger.info("Building tokenizer...")
    tokenizer = KmerTokenizer(k=config['preprocessing']['kmer_size'],stride =  config['preprocessing']['stride'])
    
    # Update vocab size in config
    if hasattr(tokenizer, 'vocab'):
        config['model']['vocab_size'] = len(tokenizer.vocab)
        logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Determine if hierarchical or single-rank
    is_hierarchical = config['model'].get('classification_type', 'single') == 'hierarchical'
    taxonomic_levels = config['model'].get('taxonomic_levels', ['species'])
    
    # For single-rank, determine which level
    if not is_hierarchical and len(taxonomic_levels) == 1:
        target_level = taxonomic_levels[0]
    else:
        target_level = None
    
    # Convert encoder dicts to LabelEncoder objects if they exist
    if encoder_dicts is not None:
        if is_hierarchical:
            # Create LabelEncoder objects for each level
            label_encoders = {
                level: LabelEncoder.from_dict(encoder_dicts[level])
                for level in taxonomic_levels
                if level in encoder_dicts
            }
        else:
            # For single-rank, just get the encoder for the target level
            if target_level and target_level in encoder_dicts:
                label_encoders = LabelEncoder.from_dict(encoder_dicts[target_level])
            elif 'species' in encoder_dicts:
                label_encoders = LabelEncoder.from_dict(encoder_dicts['species'])
    
    # Create data loaders
    if is_hierarchical:
        # Use DataFrames directly for hierarchical
        train_loader, val_loader = create_data_loaders(
            train_df, val_df, tokenizer,
            batch_size=config['training']['batch_size'],
            max_length=config['preprocessing']['max_length'],
            num_workers=config['training']['num_workers'],
            hierarchical=True,
            taxonomic_levels=taxonomic_levels,
            label_encoders=label_encoders  # Pass pre-built encoders
        )
        # Get num_classes from dataset
        num_classes = train_loader.dataset.num_classes_per_level
    else:
        # Prepare data for single-rank
        if target_level and target_level != 'species':
            # For non-species single ranks, use the specific column
            train_data = {
                'sequences': train_df['sequence'].tolist(),
                'labels': train_df[target_level].tolist(),
                'num_sequences': len(train_df),
                'num_classes': train_df[target_level].nunique()
            }
            val_data = {
                'sequences': val_df['sequence'].tolist(),
                'labels': val_df[target_level].tolist(),
                'num_sequences': len(val_df),
                'num_classes': val_df[target_level].nunique()
            }
        else:
            # Default species-level (genus_species)
            train_data = prepare_data_for_training(train_df)
            val_data = prepare_data_for_training(val_df)
        
        train_loader, val_loader = create_data_loaders(
            train_data, val_data, tokenizer,
            batch_size=config['training']['batch_size'],
            max_length=config['preprocessing']['max_length'],
            num_workers=config['training']['num_workers'],
            hierarchical=False,
            label_encoder=label_encoders  # Pass single pre-built encoder
        )
        num_classes = train_loader.dataset.num_classes
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        vocab_size=config['model']['vocab_size'],
        num_classes=num_classes,
        config=config
    )
    
    # Create output directory in the experiment folder structure
    if is_hierarchical:
        model_type = "hierarchical"
    else:
        model_type = f"single_{target_level}" if target_level else "single_species"
    
    # Models go under experiments/exp_type/dataset/models/union_type/fold_N/model_type
    output_dir = experiment_base / 'models' / config['experiment']['union_type'] / f"fold_{fold}" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata at experiment level (only once)
    experiment_info_path = experiment_base / 'experiment_info.json'
    if not experiment_info_path.exists():
        experiment_info = {
            'experiment_type': config['experiment']['fold_type'],
            'dataset_size': config['experiment']['dataset_size'],
            'union_types': ['standard', 'conservative'],
            'num_folds': config['experiment']['num_folds'],
            'created': datetime.now().isoformat(),
            'description': f"K-fold cross-validation experiment with {config['experiment']['fold_type'].replace('_', ' ')}"
        }
        with open(experiment_info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
    
    # Save config in model directory
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    if is_hierarchical:
        # Get label encoders and vocab for hierarchical
        label_encoders = train_loader.dataset.label_encoders if hasattr(train_loader.dataset, 'label_encoders') else None
        vocab = tokenizer.vocab if hasattr(tokenizer, 'vocab') else None
        
        trainer = HierarchicalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=output_dir,
            fold=fold,
            taxonomic_levels=taxonomic_levels,
            label_encoders=label_encoders,
            vocab=vocab,
            l1_lambda=config['model'].get('l1_lambda', 1e-4),
            use_uncertainty_weighting=config['model'].get('use_uncertainty_weighting', False)
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=output_dir,
            fold=fold
        )
    
    # Train
    start_time = datetime.now()
    results = trainer.train()
    end_time = datetime.now()
    
    # Save results
    results['training_time'] = str(end_time - start_time)
    results['timestamp'] = datetime.now().isoformat()
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete for fold {fold}")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    logger.info(f"Training time: {results['training_time']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a single fold")
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-10)')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config, paths = load_config()
    
    # Override config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
            config.update(custom_config)
    
    # Update seed
    config['seed'] = args.seed
    
    # Train fold
    results = train_fold(args.fold, config, paths)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Training Complete for Fold {args.fold}")
    print("="*50)
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Training Time: {results['training_time']}")
    print(f"Results saved to: {paths['models_dir']}")


if __name__ == "__main__":
    main()