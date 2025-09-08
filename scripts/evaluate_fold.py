#!/usr/bin/env python3
"""
Evaluate a trained model on a single fold
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
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_fold_data, prepare_data_for_training, create_data_loaders, LabelEncoder
from src.preprocessing import KmerTokenizer
from src.model import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    """Load configuration from config file and environment"""
    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env.example'
    load_dotenv(env_path)
    
    # Load config
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


def load_trained_model(checkpoint_path: Path, config: dict, num_classes, vocab_size: int, use_best: bool = False):
    """Load model from checkpoint"""
    checkpoint_dir = Path(checkpoint_path)
    
    # Find checkpoint file
    if use_best:
        # Look for best checkpoint in results.json
        results_path = checkpoint_dir / 'results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            best_epoch = results.get('best_epoch', 1)  # Default to epoch 1 if not found
            checkpoint_file = checkpoint_dir / f'checkpoint_epoch_{best_epoch}.pt'
        else:
            logger.warning("results.json not found, using latest checkpoint")
            checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoint_files:
                checkpoint_file = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
            else:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    else:
        # Use latest checkpoint
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoint_files:
            checkpoint_file = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    
    # Create model architecture
    model = create_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        config=config
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def get_model_type_from_config(config: dict) -> str:
    """Determine model type from config"""
    is_hierarchical = config['model'].get('classification_type', 'single') == 'hierarchical'
    if is_hierarchical:
        return 'hierarchical'
    else:
        # For single-rank, determine which level
        taxonomic_levels = config['model'].get('taxonomic_levels', ['species'])
        if len(taxonomic_levels) == 1:
            return f'single_{taxonomic_levels[0]}'
        else:
            return 'single_species'  # Default


def run_inference(model, data_loader, device, is_hierarchical: bool, taxonomic_levels: list = None):
    """Run inference on validation data"""
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference"):
            if batch is None:
                continue
                
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            if is_hierarchical:
                # logits is a list of tensors for each level
                probabilities = [torch.softmax(level_logits, dim=-1) for level_logits in logits]
                predictions = [torch.argmax(level_logits, dim=-1) for level_logits in logits]
                
                # Process each sample in batch
                for i in range(len(input_ids)):
                    sample_result = {}
                    
                    # Get true labels
                    for j, level in enumerate(taxonomic_levels):
                        if 'output' in batch and level in batch['output']:
                            sample_result[f'true_{level}'] = batch['output'][level]['encoded_label'][i].item()
                        
                        # Get predictions and probabilities
                        sample_result[f'pred_{level}'] = predictions[j][i].item()
                        
                        # Get top-5 probabilities with indices
                        level_probs = probabilities[j][i]
                        top5_probs, top5_indices = torch.topk(level_probs, min(5, len(level_probs)))
                        sample_result[f'prob_{level}_top5'] = list(zip(
                            top5_probs.cpu().numpy().astype(str),
                            top5_indices.cpu().numpy().astype(str)
                        ))
                    
                    results.append(sample_result)
            else:
                # Single-rank model
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                level = taxonomic_levels[0] if taxonomic_levels else 'species'
                
                # Process each sample in batch
                for i in range(len(input_ids)):
                    sample_result = {}
                    
                    # Get true label
                    if 'label' in batch:
                        sample_result[f'true_{level}'] = batch['label'][i].item()
                    
                    # Get prediction
                    sample_result[f'pred_{level}'] = predictions[i].item()
                    
                    # Get top-5 probabilities
                    sample_probs = probabilities[i]
                    top5_probs, top5_indices = torch.topk(sample_probs, min(5, len(sample_probs)))
                    sample_result[f'prob_{level}_top5'] = list(zip(
                        top5_probs.cpu().numpy().astype(str),
                        top5_indices.cpu().numpy().astype(str)
                    ))
                    
                    results.append(sample_result)
    
    return results


def evaluate_fold(fold: int, config: dict, paths: dict, use_best: bool = False, output_dir: str = None):
    """Main evaluation function"""
    logger.info(f"Starting evaluation for fold {fold}")
    
    # Build experiment base path (reuse train_fold logic)
    experiment_base = Path(paths['experiments_dir']) / config['experiment']['fold_type'] / \
                      config['experiment']['dataset_size']
    
    # Auto-generate checkpoint path based on config
    is_hierarchical = config['model'].get('classification_type', 'single') == 'hierarchical'
    if is_hierarchical:
        model_type = "hierarchical"
    else:
        taxonomic_levels = config['model'].get('taxonomic_levels', ['species'])
        target_level = taxonomic_levels[0] if len(taxonomic_levels) == 1 else 'species'
        model_type = f"single_{target_level}"
    
    checkpoint_path = experiment_base / 'models' / config['experiment']['union_type'] / f"fold_{fold}" / model_type
    logger.info(f"Using checkpoint path: {checkpoint_path}")
    
    # Data path
    data_path = experiment_base / 'data' / f"{config['experiment']['union_type']}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    
    # Load fold data (only validation)
    _, val_df = load_fold_data(data_path, fold, config['experiment']['fold_type'])
    
    # Load label encoders
    encoder_path = data_path.parent / f"label_encoders_{config['experiment']['union_type']}_global.json"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
    
    logger.info(f"Loading label encoders from {encoder_path}")
    with open(encoder_path, 'r') as f:
        encoder_dicts = json.load(f)
    
    # Create tokenizer
    logger.info("Building tokenizer...")
    tokenizer = KmerTokenizer(k=config['preprocessing']['kmer_size'], stride=config['preprocessing']['stride'])
    
    # Update vocab size in config
    if hasattr(tokenizer, 'vocab'):
        config['model']['vocab_size'] = len(tokenizer.vocab)
    
    # Determine if hierarchical or single-rank
    is_hierarchical = config['model'].get('classification_type', 'single') == 'hierarchical'
    taxonomic_levels = config['model'].get('taxonomic_levels', ['species'])
    
    # Convert encoder dicts to LabelEncoder objects
    if is_hierarchical:
        label_encoders = {
            level: LabelEncoder.from_dict(encoder_dicts[level])
            for level in taxonomic_levels
            if level in encoder_dicts
        }
        num_classes = [len(label_encoders[level].label_to_index) for level in taxonomic_levels]
    else:
        # Single-rank
        target_level = taxonomic_levels[0] if len(taxonomic_levels) == 1 else 'species'
        if target_level in encoder_dicts:
            label_encoders = LabelEncoder.from_dict(encoder_dicts[target_level])
        else:
            label_encoders = LabelEncoder.from_dict(encoder_dicts['species'])
        num_classes = len(label_encoders.label_to_index)
    
    # Create data loader (validation only, with training=False)
    if is_hierarchical:
        _, val_loader = create_data_loaders(
            val_df, val_df, tokenizer,
            batch_size=config['training']['batch_size'],
            max_length=config['preprocessing']['max_length'],
            num_workers=config['training']['num_workers'],
            hierarchical=True,
            taxonomic_levels=taxonomic_levels,
            label_encoders=label_encoders,
            config=config
        )
    else:
        # Prepare validation data for single-rank
        val_data = prepare_data_for_training(val_df)
        # For single-rank evaluation, we only need validation loader
        # Pass dummy train data to satisfy create_data_loaders interface
        dummy_train_data = {'sequences': ['DUMMY'], 'labels': ['DUMMY']}
        _, val_loader = create_data_loaders(
            dummy_train_data, val_data, tokenizer,
            batch_size=config['training']['batch_size'],
            max_length=config['preprocessing']['max_length'],
            num_workers=config['training']['num_workers'],
            hierarchical=False,
            label_encoder=label_encoders,
            config=config
        )
    
    # Load trained model
    model, checkpoint = load_trained_model(
        checkpoint_path, config, num_classes, 
        config['model']['vocab_size'], use_best
    )
    
    # Run inference
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running inference on device: {device}")
    
    inference_results = run_inference(
        model, val_loader, device, is_hierarchical, taxonomic_levels
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(inference_results)
    df['fold'] = fold
    df['model_type'] = get_model_type_from_config(config)
    
    # Add sequence IDs if available
    if len(df) == len(val_df):
        df['sequence_id'] = val_df['sequence_id'].values
    
    # Convert label indices back to names for readability
    if is_hierarchical:
        for level in taxonomic_levels:
            if level in encoder_dicts:
                encoder = encoder_dicts[level]
                # Convert true labels
                if f'true_{level}' in df.columns:
                    df[f'true_{level}_name'] = df[f'true_{level}'].apply(
                        lambda x: encoder['index_to_label'].get(str(x), 'unknown')
                    )
                # Convert predicted labels
                if f'pred_{level}' in df.columns:
                    df[f'pred_{level}_name'] = df[f'pred_{level}'].apply(
                        lambda x: encoder['index_to_label'].get(str(x), 'unknown')
                    )
                # Convert top-5 indices to names
                if f'prob_{level}_top5' in df.columns:
                    df[f'prob_{level}_top5'] = df[f'prob_{level}_top5'].apply(
                        lambda top5: [[prob, encoder['index_to_label'].get(str(idx), 'unknown')] 
                                     for prob, idx in top5]
                    )
    else:
        level = taxonomic_levels[0] if taxonomic_levels else 'species'
        encoder = encoder_dicts.get(level, encoder_dicts.get('species', {}))
        
        if f'true_{level}' in df.columns:
            df[f'true_{level}_name'] = df[f'true_{level}'].apply(
                lambda x: encoder.get('index_to_label', {}).get(str(x), 'unknown')
            )
        if f'pred_{level}' in df.columns:
            df[f'pred_{level}_name'] = df[f'pred_{level}'].apply(
                lambda x: encoder.get('index_to_label', {}).get(str(x), 'unknown')
            )
        if f'prob_{level}_top5' in df.columns:
            df[f'prob_{level}_top5'] = df[f'prob_{level}_top5'].apply(
                lambda top5: [[prob, encoder.get('index_to_label', {}).get(str(idx), 'unknown')] 
                             for prob, idx in top5]
            )
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(checkpoint_path).parent
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_type = get_model_type_from_config(config)
    output_file = output_path / f'evaluation_results_fold_{fold}_{model_type}.parquet'
    
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved evaluation results to {output_file}")
    logger.info(f"Results shape: {df.shape}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a single fold")
    parser.add_argument('--fold', type=int, required=True, help='Fold number (1-10)')
    parser.add_argument('--mode', choices=['final', 'best'], default='final', help='Use final epoch or best epoch checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config, paths = load_config(args.config)
    
    # Update seed
    config['seed'] = args.seed
    
    # Run evaluation
    use_best = (args.mode == 'best')
    results = evaluate_fold(
        args.fold, config, paths, 
        use_best, args.output_dir
    )
    
    print(f"\nEvaluation complete for fold {args.fold}")
    print(f"Results saved with {len(results)} samples")


if __name__ == "__main__":
    main()