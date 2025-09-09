#!/usr/bin/env python3
"""
Performance Analysis Script for 10-Fold Cross-Validation Results
Generates heatmaps, statistical comparisons, and lineage validation analysis.

Usage:
    python scripts/analyze_performance.py --base-path /path/to/experiment/models/standard
    python scripts/analyze_performance.py --base-path /path/to/experiment/models/standard --lineage-ref /path/to/experiment/data
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import ttest_rel
import logging
import pickle

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

def get_fixed_thresholds(level):
    """Get interpretable fixed thresholds based on taxonomic level"""
    # Use meaningful, interpretable thresholds
    if level in ['phylum', 'class']:
        return [50, 70, 85, 95, 100]  # Very high accuracy expected
    elif level in ['order', 'family']:
        return [30, 50, 70, 85, 100]  # Moderate accuracy expected  
    else:  # genus, species
        return [5, 15, 30, 50, 100]   # Lower accuracy expected
        
def value_to_symbol(value, thresholds):
    """Convert accuracy value to visual symbol based on fixed thresholds"""
    if np.isnan(value):
        return '?'
    
    if value >= thresholds[4]:
        return '■'  # Excellent (>95%, >85%, >50%)
    elif value >= thresholds[3]:
        return '▓'  # Very Good (85-95%, 70-85%, 30-50%) 
    elif value >= thresholds[2]:
        return '▒'  # Good (70-85%, 50-70%, 15-30%)
    elif value >= thresholds[1]:
        return '░'  # Fair (50-70%, 30-50%, 5-15%)
    else:
        return '·'  # Poor (<50%, <30%, <5%)

def load_lineage_reference(lineage_ref_dir):
    """Load lineage reference set for validation"""
    lineage_ref_path = Path(lineage_ref_dir)
    pickle_path = lineage_ref_path / "valid_lineages_set.pkl"
    
    if not pickle_path.exists():
        logger.warning(f"Lineage reference not found: {pickle_path}")
        return None, []
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            return data['lineage_set'], data['taxonomic_levels']
    except Exception as e:
        logger.error(f"Failed to load lineage reference: {e}")
        return None, []

def calculate_lineage_validity(df, valid_lineages_set, taxonomic_levels):
    """Calculate what percentage of predictions form valid lineages"""
    if valid_lineages_set is None:
        return np.nan, np.nan
    
    total_predictions = len(df)
    if total_predictions == 0:
        return np.nan, np.nan
    
    valid_count = 0
    accurate_count = 0
    
    for _, row in df.iterrows():
        # Get predicted lineage
        pred_lineage = tuple(row[f'pred_{level}_name'] for level in taxonomic_levels)
        # Get true lineage  
        true_lineage = tuple(row[f'true_{level}_name'] for level in taxonomic_levels)
        
        # Check if predicted lineage is valid
        if pred_lineage in valid_lineages_set:
            valid_count += 1
        
        # Check if predicted lineage is exactly correct
        if pred_lineage == true_lineage:
            accurate_count += 1
    
    validity_rate = valid_count / total_predictions
    accuracy_rate = accurate_count / total_predictions
    
    return validity_rate, accuracy_rate

def print_performance_heatmap(results_dict, taxonomic_levels):
    """Print performance heatmap with interpretable thresholds"""
    
    print("\nPERFORMANCE HEATMAP (Accuracy %)")
    print("■ = Excellent  ▓ = Very Good  ▒ = Good  ░ = Fair  · = Poor")
    print("Thresholds are fixed per taxonomic level for interpretability")
    print()
    
    # Header - simplified numbering for better alignment
    header = "       " + "".join(f"{f:2d} " for f in range(1, 11)) + " Mean±Std   Δ    Sig"
    print(header)
    print("     ┌" + "─" * (len(header) - 6) + "┐")
    
    for level in taxonomic_levels:
        thresholds = get_fixed_thresholds(level)
        
        # Get values for this level
        all_hier_values = [results_dict[f][level]['hierarchical'] for f in range(1, 11)]
        all_ensemble_values = [results_dict[f][level]['single_ensemble'] for f in range(1, 11)]
        
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

def print_lineage_validation(lineage_results):
    """Print lineage validation analysis"""
    
    print("\nLINEAGE VALIDATION ANALYSIS")
    print("=" * 80)
    
    if not lineage_results:
        print("⚠️  No lineage validation data available")
        print("   Run: python scripts/create_lineage_reference.py --data-path /path/to/data.csv --output-dir /path/to/output")
        return
    
    # Summary table - compact format with H/S columns
    print("Taxonomic Consistency Metrics:")
    print("─" * 80)
    header = f"{'Fold':<6} │ {'Valid Lineages':<19} │ {'Perfect Lineages':<19} │ {'Samples':<8}"
    print(header)
    sub_header = f"{'':>6} │ {'H':>8} {'S':>8} │ {'H':>8} {'S':>8} │"
    print(sub_header)
    print("─" * max(len(header), len(sub_header)))
    
    for fold in sorted(lineage_results.keys()):
        hier_data = lineage_results[fold].get('hierarchical', (np.nan, np.nan, 0))
        ens_data = lineage_results[fold].get('single_ensemble', (np.nan, np.nan, 0))
        
        hier_validity, hier_accuracy, hier_samples = hier_data
        ens_validity, ens_accuracy, ens_samples = ens_data
        
        samples = max(hier_samples, ens_samples)  # Should be same, but just in case
        
        hier_val_str = f"{hier_validity*100:5.1f}%" if not np.isnan(hier_validity) else "  N/A"
        ens_val_str = f"{ens_validity*100:5.1f}%" if not np.isnan(ens_validity) else "  N/A"
        hier_acc_str = f"{hier_accuracy*100:5.1f}%" if not np.isnan(hier_accuracy) else "  N/A"
        ens_acc_str = f"{ens_accuracy*100:5.1f}%" if not np.isnan(ens_accuracy) else "  N/A"
        
        print(f"F{fold:<5} │ {hier_val_str:>8} {ens_val_str:>8} │ {hier_acc_str:>8} {ens_acc_str:>8} │ {samples:<8}")
    
    # Summary statistics
    print("\n" + "─" * 80)
    print("OVERALL SUMMARY:")
    
    # Calculate means
    hier_validities = []
    hier_accuracies = []
    ens_validities = []
    ens_accuracies = []
    
    for fold in lineage_results:
        if 'hierarchical' in lineage_results[fold]:
            v, a, _ = lineage_results[fold]['hierarchical']
            if not np.isnan(v):
                hier_validities.append(v * 100)
                hier_accuracies.append(a * 100)
        
        if 'single_ensemble' in lineage_results[fold]:
            v, a, _ = lineage_results[fold]['single_ensemble']
            if not np.isnan(v):
                ens_validities.append(v * 100)
                ens_accuracies.append(a * 100)
    
    if hier_validities and ens_validities:
        print(f"Valid Lineages     - Hierarchical:     {np.mean(hier_validities):5.1f}±{np.std(hier_validities):4.1f}%")
        print(f"                   - Single Ensemble: {np.mean(ens_validities):5.1f}±{np.std(ens_validities):4.1f}%")
        print()
        print(f"Perfect Lineages   - Hierarchical:     {np.mean(hier_accuracies):5.1f}±{np.std(hier_accuracies):4.1f}%")
        print(f"                   - Single Ensemble: {np.mean(ens_accuracies):5.1f}±{np.std(ens_accuracies):4.1f}%")
        
        # Statistical tests
        if len(hier_validities) >= 3 and len(ens_validities) >= 3:
            try:
                _, p_val_validity = ttest_rel(hier_validities, ens_validities)
                _, p_val_accuracy = ttest_rel(hier_accuracies, ens_accuracies)
                
                print(f"\nStatistical Significance:")
                print(f"Valid Lineages Difference: p = {p_val_validity:.4f}")
                print(f"Perfect Lineages Difference: p = {p_val_accuracy:.4f}")
            except:
                pass
    
    print("\n📊 Key Insights:")
    print("• Valid Lineages: % of predictions that form biologically consistent taxonomic paths")
    print("• Perfect Lineages: % of predictions where entire taxonomic path is exactly correct")
    print("• Higher validity indicates better taxonomic consistency")
    print("• Perfect lineage rate should match overall species accuracy")

def analyze_confusion_by_level(results_dict, taxonomic_levels, base_path):
    """Analyze most confused labels and their top misclassifications per level"""
    
    print("\nCONFUSION ANALYSIS BY TAXONOMIC LEVEL")
    print("=" * 80)
    
    for level in taxonomic_levels:
        print(f"\n{level.upper()} LEVEL:")
        print("─" * 80)
        
        for model_type in ['hierarchical', 'single_ensemble']:
            model_name = "Hierarchical Model" if model_type == 'hierarchical' else "Single Ensemble Model"
            print(f"{model_name}:")
            
            # Collect all predictions across folds for this model type and level
            all_true_labels = []
            all_pred_labels = []
            
            for fold in results_dict.keys():
                # Load the actual parquet file to get individual predictions
                parquet_file = base_path / f"fold_{fold}" / f"evaluation_results_fold_{fold}_{model_type}.parquet"
                
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        true_col = f'true_{level}_name'
                        pred_col = f'pred_{level}_name'
                        
                        if true_col in df.columns and pred_col in df.columns:
                            all_true_labels.extend(df[true_col].tolist())
                            all_pred_labels.extend(df[pred_col].tolist())
                    except Exception as e:
                        logger.warning(f"Failed to load {parquet_file}: {e}")
                        continue
            
            if not all_true_labels:
                print(f"  No data available for {model_type} at {level} level")
                continue
            
            # Find the most confused true label (lowest accuracy)
            confusion_data = {}
            for true_label, pred_label in zip(all_true_labels, all_pred_labels):
                if true_label not in confusion_data:
                    confusion_data[true_label] = {'correct': 0, 'total': 0, 'misclassifications': {}}
                
                confusion_data[true_label]['total'] += 1
                if true_label == pred_label:
                    confusion_data[true_label]['correct'] += 1
                else:
                    # Track misclassification
                    if pred_label not in confusion_data[true_label]['misclassifications']:
                        confusion_data[true_label]['misclassifications'][pred_label] = 0
                    confusion_data[true_label]['misclassifications'][pred_label] += 1
            
            # Find the label with lowest accuracy (most confused)
            most_confused_label = None
            lowest_accuracy = 1.0
            
            for true_label, data in confusion_data.items():
                if data['total'] >= 10:  # Only consider labels with at least 10 samples
                    accuracy = data['correct'] / data['total']
                    if accuracy < lowest_accuracy:
                        lowest_accuracy = accuracy
                        most_confused_label = true_label
            
            if most_confused_label is None:
                print("  No sufficiently frequent labels found")
                print()
                continue
            
            # Get top-5 misclassifications for the most confused label
            misclassifications = confusion_data[most_confused_label]['misclassifications']
            total_samples = confusion_data[most_confused_label]['total']
            
            # Sort misclassifications by frequency
            sorted_misclassifications = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
            top5_misclassifications = sorted_misclassifications[:5]
            
            print(f"  True Label: {most_confused_label:<20} (accuracy: {lowest_accuracy*100:5.1f}%)")
            
            for i, (pred_label, count) in enumerate(top5_misclassifications, 1):
                percentage = (count / total_samples) * 100
                print(f"              {i}. {pred_label:<20}: {count:3d} sequences ({percentage:4.1f}%)")
            
            print()

def calculate_confusion_from_parquets(base_path, taxonomic_levels):
    """Calculate confusion matrices from saved parquet files"""
    base_path = Path(base_path)
    
    # This function would be called if we need to analyze confusion patterns
    # For now, we'll embed the logic in the main analysis function
    pass

def main():
    parser = argparse.ArgumentParser(description='Analyze 10-fold cross-validation performance')
    parser.add_argument('--base-path', required=True, type=str,
                      help='Base path to experiment models directory (e.g., /path/to/models/standard)')
    parser.add_argument('--lineage-ref', type=str,
                      help='Path to lineage reference directory (optional, enables lineage validation)')
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
    
    # Load lineage reference if provided
    valid_lineages_set = None
    lineage_taxonomic_levels = []
    if args.lineage_ref:
        valid_lineages_set, lineage_taxonomic_levels = load_lineage_reference(args.lineage_ref)
        if valid_lineages_set:
            logger.info(f"Loaded {len(valid_lineages_set)} valid lineages for validation")
    
    # Load all results
    results_dict = {}
    lineage_results = {}
    taxonomic_levels = args.levels
    
    for fold in args.folds:
        hier_df, ensemble_df = load_results(base_path, fold)
        
        if hier_df is None or ensemble_df is None:
            logger.warning(f"Skipping fold {fold} due to missing files")
            continue
        
        results_dict[fold] = {}
        lineage_results[fold] = {}
        
        for level in taxonomic_levels:
            hier_acc = calculate_accuracy(hier_df, level)
            ensemble_acc = calculate_accuracy(ensemble_df, level)
            
            results_dict[fold][level] = {
                'hierarchical': hier_acc * 100,  # Convert to percentage
                'single_ensemble': ensemble_acc * 100
            }
        
        # Calculate lineage validation if reference is available
        if valid_lineages_set and lineage_taxonomic_levels:
            # Hierarchical lineage validation
            hier_validity, hier_accuracy = calculate_lineage_validity(
                hier_df, valid_lineages_set, lineage_taxonomic_levels
            )
            lineage_results[fold]['hierarchical'] = (hier_validity, hier_accuracy, len(hier_df))
            
            # Ensemble lineage validation
            ens_validity, ens_accuracy = calculate_lineage_validity(
                ensemble_df, valid_lineages_set, lineage_taxonomic_levels
            )
            lineage_results[fold]['single_ensemble'] = (ens_validity, ens_accuracy, len(ensemble_df))
    
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
    
    # Print lineage validation if available
    if lineage_results:
        print_lineage_validation(lineage_results)
    
    # Print confusion analysis
    analyze_confusion_by_level(results_dict, taxonomic_levels, base_path)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")

if __name__ == "__main__":
    main()