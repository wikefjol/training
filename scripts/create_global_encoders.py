#!/usr/bin/env python3
"""
Create global label encoders for k-fold cross-validation

This script creates ONE label encoder per taxonomic level that contains ALL labels
from the complete dataset. This ensures all folds use identical model architectures.
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from data import LabelEncoder

def create_global_encoders(data_path: str, output_path: str):
    """
    Create global label encoders from complete dataset
    
    Args:
        data_path: Path to complete dataset CSV
        output_path: Path to save global encoders JSON
    """
    # Read complete dataset
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} sequences from {data_path}")
    
    # Taxonomic levels to encode
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # Create encoders for each level
    global_encoders = {}
    
    for level in taxonomic_levels:
        if level not in df.columns:
            print(f"Warning: {level} not found in data, skipping")
            continue
            
        # Get all unique labels for this level (remove NaN)
        unique_labels = sorted(df[level].dropna().unique())
        print(f"{level}: {len(unique_labels)} unique labels")
        
        # Create encoder
        encoder = LabelEncoder(unique_labels)
        
        # Convert to serializable format
        global_encoders[level] = {
            'label_to_index': encoder.label_to_index,
            'index_to_label': encoder.index_to_label
        }
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(global_encoders, f, indent=2)
    
    print(f"Saved global encoders to {output_path}")
    
    # Print summary
    print("\\nEncoder Summary:")
    for level, encoder_data in global_encoders.items():
        print(f"  {level}: {len(encoder_data['label_to_index'])} classes")

def main():
    parser = argparse.ArgumentParser(description="Create global label encoders")
    parser.add_argument('--data', required=True, help='Path to complete dataset CSV')
    parser.add_argument('--output', required=True, help='Path to save global encoders JSON')
    
    args = parser.parse_args()
    
    create_global_encoders(args.data, args.output)

if __name__ == "__main__":
    main()