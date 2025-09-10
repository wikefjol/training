#!/bin/env bash
# Usage: sbatch finetune_model.sh <config_file> <hours>
# Example: sbatch finetune_model.sh configs/finetuning_5mer_config.json 12

#SBATCH -A NAISS2025-22-110
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 0-16:00:00
#SBATCH -J ft_model
#SBATCH -o /cephyr/users/filbern/Alvis/workspace/training/sbatch_logs/finetune_test_hierarchical/%x_%j.out

# Load modules
ml purge
ml load virtualenv/20.23.1-GCCcore-12.3.0
ml load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
ml load Transformers/4.39.3-gfbf-2023a
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load scikit-learn/1.3.1
ml load matplotlib/3.7.2-gfbf-2023a
ml load h5py/3.9.0-foss-2023a
ml load IPython/8.14.0-GCCcore-12.3.0
ml load JupyterLab/4.0.5-GCCcore-12.3.0
ml load Pillow/10.0.0-GCCcore-12.3.0
ml load plotly.py/5.16.0-GCCcore-12.3.0
ml load Seaborn/0.13.2-gfbf-2023a

# Activate virtual environment
source /mimer/NOBACKUP/groups/snic2022-22-552/filbern/environment/venv/bin/activate

# Change to working directory
cd /cephyr/users/filbern/Alvis/workspace/training

echo "Starting hierarchical model training test at: $(date)"
echo "Testing fold 1 with full-scale hierarchical config"

# Train the hierarchical model for fold 1
python scripts/train_fold.py --fold 1 --config configs/hierarchical.yaml

echo "Training completed at: $(date)"

# If training succeeds, run evaluation to complete the full pipeline timing
if [ $? -eq 0 ]; then
    echo "Training succeeded. Starting evaluation..."
    python scripts/evaluate_fold.py --fold 1 --config configs/hierarchical.yaml --mode best
    
    if [ $? -eq 0 ]; then
        echo "Full pipeline completed successfully at: $(date)"
    else
        echo "Evaluation failed"
        exit 1
    fi
else
    echo "Training failed"
    exit 1
fi