#!/bin/bash

# Generate parallel job commands with dependencies
# Usage: ./generate_jobs.sh

MODELS=(
    "tiny_hierarchical"
    "tiny_single_phylum" 
    "tiny_single_class"
    "tiny_single_order"
    "tiny_single_family"
    "tiny_single_genus"
    "tiny_single_species"
)

FOLDS=(2 3 4 5 6 7 8 9 10)  # Skip fold 1 since it's already done

BASE_DIR="/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/experiments/exp1_sequence_fold/debug_5genera_10fold/models/standard"

# Create directories
mkdir -p logs/{train,eval,agg}

# Generate training commands
echo "# Training commands" > train_cmds.txt
for fold in "${FOLDS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "python scripts/train_fold.py --fold $fold --config configs/$model.yaml" >> train_cmds.txt
    done
done

# Generate evaluation commands (depend on training)
echo "# Evaluation commands" > eval_cmds.txt  
for fold in "${FOLDS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "python scripts/evaluate_fold.py --fold $fold --config configs/$model.yaml --mode best" >> eval_cmds.txt
    done
done

# Generate aggregation commands (depend on all single-rank evals for that fold)
echo "# Aggregation commands" > agg_cmds.txt
for fold in "${FOLDS[@]}"; do
    echo "python scripts/aggregate_single_rank_results.py --fold $fold --input-dir $BASE_DIR/fold_$fold/ --output-dir $BASE_DIR/fold_$fold/" >> agg_cmds.txt
done

echo "Generated job files:"
echo "- train_cmds.txt ($(wc -l < train_cmds.txt) training jobs)"
echo "- eval_cmds.txt ($(wc -l < eval_cmds.txt) evaluation jobs)" 
echo "- agg_cmds.txt ($(wc -l < agg_cmds.txt) aggregation jobs)"