#!/bin/bash

# Run the complete training/evaluation pipeline with dependencies
# Usage: ./run_parallel_pipeline.sh

set -e  # Exit on any error

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting parallel pipeline..."

# Create log directories
mkdir -p logs/{train,eval,agg}

# Generate job commands
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating job commands..."
./generate_jobs.sh

# Phase 1: Run all training jobs in parallel
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1: Training models..."
parallel -j6 --delay 0.2 --joblog logs/train/parallel.log \
  --results logs/train/results --retries 1 --timeout 1800 \
  'echo "[$(date +%H:%M:%S)] TRAIN: {}"; {}' :::: train_cmds.txt | tee logs/train/run.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete. Starting evaluation..."

# Phase 2: Run all evaluation jobs in parallel
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2: Evaluating models..."
parallel -j6 --delay 0.2 --joblog logs/eval/parallel.log \
  --results logs/eval/results --retries 1 --timeout 600 \
  'echo "[$(date +%H:%M:%S)] EVAL: {}"; {}' :::: eval_cmds.txt | tee logs/eval/run.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation complete. Starting aggregation..."

# Phase 3: Run aggregation jobs sequentially (they're quick)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 3: Aggregating results..."
parallel -j3 --delay 0.5 --joblog logs/agg/parallel.log \
  --results logs/agg/results --retries 1 --timeout 300 \
  'echo "[$(date +%H:%M:%S)] AGG: {}"; {}' :::: agg_cmds.txt | tee logs/agg/run.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete!"

# Summary
echo ""
echo "=== PIPELINE SUMMARY ==="
echo "Training jobs: $(wc -l < train_cmds.txt)"
echo "Evaluation jobs: $(wc -l < eval_cmds.txt)" 
echo "Aggregation jobs: $(wc -l < agg_cmds.txt)"
echo "Check logs/ directory for detailed results"