#!/usr/bin/env python3
"""
Create Lineage Reference Dataset
Extracts all valid taxonomic lineages from the complete training dataset for lineage validation.

Usage:
    python scripts/create_lineage_reference.py --data-path /path/to/experiment/data/standard.csv --output-dir /path/to/experiment/data/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_lineage_reference(data_path: Path, output_dir: Path):
    """
    Create reference dataset of all valid taxonomic lineages from training data
    
    Args:
        data_path: Path to the complete training dataset CSV
        output_dir: Directory to save the lineage reference files
    """
    
    logger.info(f"Loading training data from: {data_path}")
    
    # Load the complete dataset
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} sequences")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Define taxonomic levels in order
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # Check required columns exist
    missing_cols = [col for col in taxonomic_levels if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)
    
    logger.info(f"Found taxonomic levels: {taxonomic_levels}")
    
    # Remove rows with any missing taxonomic information
    before_cleanup = len(df)
    df_clean = df[taxonomic_levels].dropna()
    after_cleanup = len(df_clean)
    
    logger.info(f"Removed {before_cleanup - after_cleanup} rows with missing taxonomic data")
    logger.info(f"Working with {after_cleanup} complete lineages")
    
    # Create lineage DataFrame with unique combinations
    logger.info("Creating lineage reference dataset...")
    lineage_df = df_clean[taxonomic_levels].drop_duplicates().reset_index(drop=True)
    
    logger.info(f"Found {len(lineage_df)} unique valid lineages")
    
    # Print summary statistics
    print("\nLINEAGE REFERENCE SUMMARY")
    print("=" * 50)
    print(f"Total sequences in dataset: {before_cleanup:,}")
    print(f"Complete lineages: {after_cleanup:,}")
    print(f"Unique valid lineages: {len(lineage_df):,}")
    print()
    
    # Show diversity at each taxonomic level
    print("Taxonomic Diversity:")
    for level in taxonomic_levels:
        unique_count = lineage_df[level].nunique()
        print(f"  {level.title():<8}: {unique_count:,} unique taxa")
    
    # Show example lineages
    print(f"\nExample Valid Lineages (showing first 5):")
    print("-" * 80)
    for i, row in lineage_df.head().iterrows():
        lineage_path = " → ".join([row[level] for level in taxonomic_levels])
        print(f"{i+1:2d}. {lineage_path}")
    
    if len(lineage_df) > 5:
        print(f"... and {len(lineage_df) - 5:,} more lineages")
    
    # Save lineage reference files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / "valid_lineages_reference.csv"
    lineage_df.to_csv(csv_path, index=False)
    logger.info(f"Saved lineage reference CSV: {csv_path}")
    
    # Save as parquet for faster loading
    parquet_path = output_dir / "valid_lineages_reference.parquet"
    lineage_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved lineage reference Parquet: {parquet_path}")
    
    # Create set of tuples for fast lookup and save as pickle
    import pickle
    lineage_tuples = set()
    for _, row in lineage_df.iterrows():
        lineage_tuple = tuple(row[level] for level in taxonomic_levels)
        lineage_tuples.add(lineage_tuple)
    
    pickle_path = output_dir / "valid_lineages_set.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'lineage_set': lineage_tuples,
            'taxonomic_levels': taxonomic_levels,
            'total_lineages': len(lineage_tuples),
            'created_from': str(data_path),
            'total_sequences': before_cleanup,
            'complete_sequences': after_cleanup
        }, f)
    
    logger.info(f"Saved lineage set pickle: {pickle_path}")
    
    print(f"\nFiles created in {output_dir}:")
    print(f"  • valid_lineages_reference.csv     - Human readable lineage table")
    print(f"  • valid_lineages_reference.parquet - Fast loading format")
    print(f"  • valid_lineages_set.pkl          - Fast lookup set for validation")
    
    return lineage_df, lineage_tuples

def main():
    parser = argparse.ArgumentParser(description='Create lineage reference dataset from training data')
    parser.add_argument('--data-path', required=True, type=str,
                      help='Path to complete training dataset CSV file')
    parser.add_argument('--output-dir', required=True, type=str,
                      help='Output directory for lineage reference files')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)
    
    if not data_path.suffix.lower() == '.csv':
        logger.error(f"Data file must be CSV format: {data_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Create lineage reference
    lineage_df, lineage_set = create_lineage_reference(data_path, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ Lineage reference creation complete!")
    print("\nTo use in analysis:")
    print(f"python scripts/analyze_performance.py --base-path /path/to/models --lineage-ref {output_dir}")

if __name__ == "__main__":
    main()