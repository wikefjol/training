#!/usr/bin/env python3
"""
Performance Analysis Script for 10-Fold Cross-Validation Results
Generates adaptive heatmaps and statistical comparisons between hierarchical and single-rank ensemble models.

Usage:
    python scripts/analyze_performance.py --base-path /path/to/experiment/models/standard
    python scripts/analyze_performance.py --base-path /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import ttest_rel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(base_path: Path, fold: int):
    """Load hierarchical and single ensemble results for a fold"""
    fold_dir = base_path / f"fold_{fold}"
    
    hier_file = fold_dir / f"evaluation_results_fold_{fold}_hierarchical.parquet"
    ensemble_file = fold_dir / f"evaluation_results_fold_{fold}_single_ensemble.parquet"
    
    if not hier_file.exists():
        logger.warning(f"Missing hierarchical file: {hier_file}")
        return None, None
        
    if not ensemble_file.exists():
        logger.warning(f"Missing ensemble file: {ensemble_file}")
        return None, None
    
    hier_df = pd.read_parquet(hier_file)
    ensemble_df = pd.read_parquet(ensemble_file)
    
    return hier_df, ensemble_df

def calculate_accuracy(df, level):
    """Calculate accuracy for a taxonomic level"""
    true_col = f'true_{level}'
    pred_col = f'pred_{level}'
    
    if true_col not in df.columns or pred_col not in df.columns:
        return np.nan
        
    return (df[true_col] == df[pred_col]).mean()

def get_data_driven_thresholds(values):
    """Get adaptive color thresholds based on data distribution"""
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN values
    
    if len(values) == 0:
        return [0, 0.25, 0.5, 0.75, 1.0]
    
    # Use quintiles for maximum resolution
    return np.percentile(values, [0, 20, 40, 60, 80, 100])

def value_to_symbol(value, thresholds):
    """Convert accuracy value to visual symbol based on thresholds"""
    if np.isnan(value):
        return '?'
    
    if value >= thresholds[4]:
        return '■'  # Top 20%
    elif value >= thresholds[3]:
        return '▓'  # 60-80%
    elif value >= thresholds[2]:
        return '▒'  # 40-60%
    elif value >= thresholds[1]:
        return '░'  # 20-40%
    else:
        return '·'  # Bottom 20%

def print_performance_heatmap(results_dict, taxonomic_levels):
    """Print adaptive performance heatmap"""
    
    print("\nPERFORMANCE HEATMAP (Accuracy %)")
    print("■ = Top 20%  ▓ = 60-80%  ▒ = 40-60%  ░ = 20-40%  · = Bottom 20%")
    print("Thresholds are adaptive per taxonomic level for maximum resolution")
    print()
    
    # Header
    header = "       " + "".join(f"F{f:2d} " for f in range(1, 11)) + " Mean±Std   Δ    Sig"
    print(header)
    print("     ┌" + "─" * (len(header) - 6) + "┐")
    
    for level in taxonomic_levels:
        # Get all values for this level to calculate thresholds
        all_hier_values = [results_dict[f][level]['hierarchical'] for f in range(1, 11)]
        all_ensemble_values = [results_dict[f][level]['single_ensemble'] for f in range(1, 11)]
        all_values = all_hier_values + all_ensemble_values
        
        thresholds = get_data_driven_thresholds(all_values)
        
        # Calculate statistics
        hier_mean = np.nanmean(all_hier_values)
        hier_std = np.nanstd(all_hier_values)
        ensemble_mean = np.nanmean(all_ensemble_values)
        ensemble_std = np.nanstd(all_ensemble_values)
        
        # Statistical test
        valid_pairs = [(h, e) for h, e in zip(all_hier_values, all_ensemble_values) 
                      if not (np.isnan(h) or np.isnan(e))]
        
        if len(valid_pairs) >= 3:  # Need at least 3 pairs for t-test
            hier_vals, ensemble_vals = zip(*valid_pairs)
            try:
                _, p_value = ttest_rel(hier_vals, ensemble_vals)
                if p_value < 0.001:
                    sig_str = "***"
                elif p_value < 0.01:
                    sig_str = "**"
                elif p_value < 0.05:
                    sig_str = "*"
                elif p_value < 0.1:
                    sig_str = "·"
                else:
                    sig_str = "ns"
            except:
                sig_str = "?"
        else:
            sig_str = "?"
        
        # Print hierarchical row
        hier_symbols = "".join(f"{value_to_symbol(results_dict[f][level]['hierarchical'], thresholds):>3}" 
                              for f in range(1, 11))
        diff = hier_mean - ensemble_mean
        print(f"{level[:3].title()} H│{hier_symbols} │{hier_mean:5.1f}±{hier_std:4.1f} {diff:+5.1f}  {sig_str:>3}")
        
        # Print ensemble row  
        ensemble_symbols = "".join(f"{value_to_symbol(results_dict[f][level]['single_ensemble'], thresholds):>3}" 
                                  for f in range(1, 11))
        print(f"    S│{ensemble_symbols} │{ensemble_mean:5.1f}±{ensemble_std:4.1f}")
        
        if level != taxonomic_levels[-1]:
            print("     ├" + "─" * (len(header) - 6) + "┤")
    
    print("     └" + "─" * (len(header) - 6) + "┘")
    print("\nLegend: H=Hierarchical, S=Single Ensemble")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, · p<0.1, ns=not significant")

def print_summary_table(results_dict, taxonomic_levels):
    """Print summary performance table"""
    
    print("\nSUMMARY PERFORMANCE TABLE")
    print("=" * 80)
    
    header = f"{'Level':<8} │ {'Hierarchical':<15} │ {'Single Ensemble':<15} │ {'Difference':<10} │ {'Significance':<12}"
    print(header)
    print("─" * len(header))
    
    for level in taxonomic_levels:
        # Calculate statistics
        hier_values = [results_dict[f][level]['hierarchical'] for f in range(1, 11)]
        ensemble_values = [results_dict[f][level]['single_ensemble'] for f in range(1, 11)]
        
        hier_mean = np.nanmean(hier_values)
        hier_std = np.nanstd(hier_values)
        ensemble_mean = np.nanmean(ensemble_values)
        ensemble_std = np.nanstd(ensemble_values)
        
        diff = hier_mean - ensemble_mean
        
        # Statistical test
        valid_pairs = [(h, e) for h, e in zip(hier_values, ensemble_values) 
                      if not (np.isnan(h) or np.isnan(e))]
        
        if len(valid_pairs) >= 3:
            hier_vals, ensemble_vals = zip(*valid_pairs)
            try:
                _, p_value = ttest_rel(hier_vals, ensemble_vals)
                sig_str = f"p={p_value:.4f}"
            except:
                sig_str = "error"
        else:
            sig_str = "insufficient data"
        
        print(f"{level.title():<8} │ {hier_mean:5.1f}±{hier_std:4.1f}%     │ {ensemble_mean:5.1f}±{ensemble_std:4.1f}%     │ {diff:+6.1f}%    │ {sig_str:<12}")

def print_trend_visualization(results_dict, taxonomic_levels):
    """Print trend visualization showing performance degradation"""
    
    print("\nTREND VISUALIZATION")
    print("=" * 50)
    
    for model_type in ['hierarchical', 'single_ensemble']:
        means = []
        for level in taxonomic_levels:
            values = [results_dict[f][level][model_type] for f in range(1, 11)]
            means.append(np.nanmean(values))
        
        model_name = "Hierarchical" if model_type == 'hierarchical' else "Single Ensemble"
        
        # Create visual bar representation
        max_mean = max(means) if not all(np.isnan(means)) else 1
        min_mean = min(means) if not all(np.isnan(means)) else 0
        
        bar_chars = []
        for mean in means:
            if np.isnan(mean):
                bar_chars.append('?')
            else:
                # Normalize to 0-6 range for visual representation
                normalized = int(6 * (mean - min_mean) / (max_mean - min_mean)) if max_mean > min_mean else 3
                bar_chars.append('▆▅▄▃▂▁'[min(normalized, 5)])
        
        bars = ''.join(bar_chars)
        degradation = means[0] - means[-1] if not (np.isnan(means[0]) or np.isnan(means[-1])) else 0
        
        print(f"{model_name:>15}: {bars}  {means[0]:5.1f}% → {means[-1]:5.1f}%  ▼{degradation:4.1f}%")
    
    # Show per-level comparison
    print("\nPer-Level Comparison:")
    max_overall = max([np.nanmean([results_dict[f][level][model_type] 
                                   for f in range(1, 11)]) 
                       for level in taxonomic_levels 
                       for model_type in ['hierarchical', 'single_ensemble']])
    
    for level in taxonomic_levels:
        hier_mean = np.nanmean([results_dict[f][level]['hierarchical'] for f in range(1, 11)])
        ensemble_mean = np.nanmean([results_dict[f][level]['single_ensemble'] for f in range(1, 11)])
        
        if not np.isnan(hier_mean) and not np.isnan(ensemble_mean):
            hier_bar_len = int(20 * hier_mean / max_overall)
            ensemble_bar_len = int(20 * ensemble_mean / max_overall)
            
            hier_bar = '█' * hier_bar_len
            ensemble_bar = '█' * ensemble_bar_len
            
            print(f"{level.title():<8} {hier_bar:<20} {hier_mean:5.1f}% vs {ensemble_bar:<20} {ensemble_mean:5.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Analyze 10-fold cross-validation performance')
    parser.add_argument('--base-path', required=True, type=str,
                      help='Base path to experiment models directory (e.g., /path/to/models/standard)')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 11)),
                      help='Folds to analyze (default: 1-10)')
    parser.add_argument('--levels', nargs='+', type=str, 
                      default=['phylum', 'class', 'order', 'family', 'genus', 'species'],
                      help='Taxonomic levels to analyze')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        sys.exit(1)
    
    logger.info(f"Loading results from: {base_path}")
    
    # Load all results
    results_dict = {}
    taxonomic_levels = args.levels
    
    for fold in args.folds:
        hier_df, ensemble_df = load_results(base_path, fold)
        
        if hier_df is None or ensemble_df is None:
            logger.warning(f"Skipping fold {fold} due to missing files")
            continue
        
        results_dict[fold] = {}
        
        for level in taxonomic_levels:
            hier_acc = calculate_accuracy(hier_df, level)
            ensemble_acc = calculate_accuracy(ensemble_df, level)
            
            results_dict[fold][level] = {
                'hierarchical': hier_acc * 100,  # Convert to percentage
                'single_ensemble': ensemble_acc * 100
            }
    
    if not results_dict:
        logger.error("No valid results found")
        sys.exit(1)
    
    logger.info(f"Successfully loaded results for {len(results_dict)} folds")
    
    # Generate visualizations
    print(f"\nPERFORMANCE ANALYSIS: {base_path.name.upper()}")
    print("=" * 80)
    print(f"Experiment path: {base_path}")
    print(f"Folds analyzed: {sorted(results_dict.keys())}")
    print(f"Taxonomic levels: {', '.join(taxonomic_levels)}")
    
    print_performance_heatmap(results_dict, taxonomic_levels)
    print_trend_visualization(results_dict, taxonomic_levels)
    print_summary_table(results_dict, taxonomic_levels)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")

if __name__ == "__main__":
    main()