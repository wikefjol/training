#!/usr/bin/env python3
"""
Aggregate single-rank model evaluation results into a single ensemble file

Combines 6 single-rank model parquet files into one ensemble file with the same
structure as the hierarchical results for comparative analysis.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_single_rank_files(fold: int, base_dir: Path = None):
    """Find all single-rank evaluation result files for a given fold"""
    if base_dir is None:
        base_dir = Path.cwd()
    
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    files = {}
    
    for level in taxonomic_levels:
        file_pattern = f"evaluation_results_fold_{fold}_single_{level}.parquet"
        file_path = base_dir / file_pattern
        
        if file_path.exists():
            files[level] = file_path
            logger.info(f"Found {level} file: {file_path}")
        else:
            logger.warning(f"Missing {level} file: {file_path}")
    
    return files


def aggregate_single_rank_results(fold: int, input_dir: Path = None, output_dir: Path = None):
    """
    Aggregate single-rank results into ensemble format
    
    Args:
        fold: Fold number
        input_dir: Directory containing single-rank parquet files
        output_dir: Directory to save aggregated results
    """
    if input_dir is None:
        input_dir = Path.cwd()
    if output_dir is None:
        output_dir = input_dir
    
    logger.info(f"Aggregating single-rank results for fold {fold}")
    
    # Find all single-rank files
    single_rank_files = find_single_rank_files(fold, input_dir)
    
    if len(single_rank_files) != 6:
        missing = set(['phylum', 'class', 'order', 'family', 'genus', 'species']) - set(single_rank_files.keys())
        raise FileNotFoundError(f"Missing single-rank files for levels: {missing}")
    
    # Load all single-rank results
    dfs = {}
    for level, file_path in single_rank_files.items():
        logger.info(f"Loading {level} results from {file_path}")
        df = pd.read_parquet(file_path)
        dfs[level] = df
        logger.info(f"  Shape: {df.shape}")
    
    # Verify all dataframes have same number of sequences
    sequence_counts = [len(df) for df in dfs.values()]
    if len(set(sequence_counts)) > 1:
        raise ValueError(f"Inconsistent sequence counts across files: {sequence_counts}")
    
    n_sequences = sequence_counts[0]
    logger.info(f"All files have {n_sequences} sequences")
    
    # Use the first dataframe as base and verify sequence IDs match
    base_df = dfs['phylum'].copy()
    for level, df in dfs.items():
        if not base_df['sequence_id'].equals(df['sequence_id']):
            raise ValueError(f"Sequence IDs don't match between phylum and {level}")
    
    # Create aggregated dataframe with hierarchical structure
    result_df = pd.DataFrame()
    
    # Copy metadata columns from base
    result_df['sequence_id'] = base_df['sequence_id']
    result_df['fold'] = base_df['fold']
    result_df['model_type'] = 'single_ensemble'  # Mark as ensemble
    
    # Aggregate all taxonomic levels
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    
    for level in taxonomic_levels:
        df = dfs[level]
        
        # Copy true/pred values (indices)
        result_df[f'true_{level}'] = df[f'true_{level}']
        result_df[f'pred_{level}'] = df[f'pred_{level}']
        
        # Copy label names
        result_df[f'true_{level}_name'] = df[f'true_{level}_name']
        result_df[f'pred_{level}_name'] = df[f'pred_{level}_name']
        
        # Copy top-5 probabilities
        result_df[f'prob_{level}_top5'] = df[f'prob_{level}_top5']
    
    # Reorder columns to match hierarchical format
    column_order = []
    
    # Add true/pred columns for each level
    for level in taxonomic_levels:
        column_order.extend([f'true_{level}', f'pred_{level}', f'prob_{level}_top5'])
    
    # Add metadata
    column_order.extend(['fold', 'model_type', 'sequence_id'])
    
    # Add name columns
    for level in taxonomic_levels:
        column_order.extend([f'true_{level}_name', f'pred_{level}_name'])
    
    # Reorder dataframe
    result_df = result_df[column_order]
    
    # Save aggregated results
    output_file = output_dir / f'evaluation_results_fold_{fold}_single_ensemble.parquet'
    result_df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved aggregated results to {output_file}")
    logger.info(f"Final shape: {result_df.shape}")
    logger.info(f"Columns: {len(result_df.columns)}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate single-rank evaluation results")
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--input-dir', type=str, help='Directory containing single-rank parquet files')
    parser.add_argument('--output-dir', type=str, help='Directory to save aggregated results')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        result = aggregate_single_rank_results(args.fold, input_dir, output_dir)
        print(f"\n✅ Successfully aggregated single-rank results for fold {args.fold}")
        print(f"📊 Final dataset: {result.shape[0]} sequences, {result.shape[1]} columns")
        print(f"🎯 Model type: {result['model_type'].iloc[0]}")
        
    except Exception as e:
        print(f"\n❌ Error aggregating results: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())