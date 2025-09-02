#!/usr/bin/env python3
"""
Train all folds for k-fold cross-validation
"""

import os
import sys
import json
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fold(fold: int, config_path: str = None, seed: int = 42) -> Dict:
    """
    Run training for a single fold
    
    Args:
        fold: Fold number
        config_path: Optional path to config file
        seed: Random seed
        
    Returns:
        Results dictionary
    """
    logger.info(f"Starting fold {fold}")
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'train_fold.py'),
        '--fold', str(fold),
        '--seed', str(seed)
    ]
    
    if config_path:
        cmd.extend(['--config', config_path])
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Fold {fold} completed successfully")
        
        # Parse output to get results
        # (In practice, we'd read from the saved results file)
        return {'fold': fold, 'status': 'success'}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Fold {fold} failed: {e.stderr}")
        return {'fold': fold, 'status': 'failed', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Train all folds")
    parser.add_argument('--folds', type=str, default='all', 
                       help='Folds to train: "all" or comma-separated list (e.g., "1,2,3")')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--parallel', action='store_true', 
                       help='Run folds in parallel (requires multiple GPUs)')
    
    args = parser.parse_args()
    
    # Determine which folds to run
    if args.folds == 'all':
        folds = list(range(1, 11))
    else:
        folds = [int(f) for f in args.folds.split(',')]
    
    logger.info(f"Will train folds: {folds}")
    
    # Track results
    all_results = []
    start_time = datetime.now()
    
    # Train each fold
    if args.parallel:
        logger.warning("Parallel training not yet implemented, running sequentially")
    
    for fold in folds:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}/{len(folds)}")
        logger.info(f"{'='*50}")
        
        results = run_fold(fold, args.config, args.seed)
        all_results.append(results)
        
        # Save intermediate results
        results_path = Path('all_folds_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Calculate summary statistics
    end_time = datetime.now()
    total_time = end_time - start_time
    
    successful_folds = [r for r in all_results if r['status'] == 'success']
    failed_folds = [r for r in all_results if r['status'] == 'failed']
    
    # Print summary
    print("\n" + "="*60)
    print("K-Fold Cross-Validation Complete")
    print("="*60)
    print(f"Total Time: {total_time}")
    print(f"Successful Folds: {len(successful_folds)}/{len(folds)}")
    if failed_folds:
        print(f"Failed Folds: {[r['fold'] for r in failed_folds]}")
    
    # Save final results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time': str(total_time),
        'num_folds': len(folds),
        'successful_folds': len(successful_folds),
        'failed_folds': len(failed_folds),
        'results': all_results
    }
    
    summary_path = Path('kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()