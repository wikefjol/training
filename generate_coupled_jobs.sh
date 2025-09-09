#!/bin/bash

# Generate coupled train+eval jobs for all folds and models
# Usage: ./generate_coupled_jobs.sh

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

echo "Generating coupled train+eval jobs..."

> coupled_jobs.txt  # Create empty file

for fold in "${FOLDS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "python scripts/train_fold.py --fold $fold --config configs/$model.yaml && python scripts/evaluate_fold.py --fold $fold --config configs/$model.yaml --mode best" >> coupled_jobs.txt
    done
done

echo "Generated $(wc -l < coupled_jobs.txt) coupled jobs in coupled_jobs.txt"
echo ""
echo "To run all jobs in parallel:"
echo "parallel -j6 --delay 0.2 --joblog parallel.log --timeout 2400 :::: coupled_jobs.txt"
echo ""
echo "Jobs per fold: 7 (1 hierarchical + 6 single-rank)"
echo "Total folds: 9 (2-10)" 
echo "Total jobs: 63"