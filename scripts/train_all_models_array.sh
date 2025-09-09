#!/bin/env bash
# SLURM array job script for training 70 models (10 folds × 7 model types)
# Usage: sbatch scripts/train_all_models_array.sh

#SBATCH -A NAISS2024-22-976
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 0-24:00:00
#SBATCH -J train_fungal
#SBATCH --array=0-69%8
#SBATCH -o /cephyr/users/filbern/Alvis/workspace/training/sbatch_logs/%x_%A_%a.out

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

# Array job indexing: prioritize fold completion
# Jobs 0-6: fold 1, jobs 7-13: fold 2, etc.
FOLD=$(( ${SLURM_ARRAY_TASK_ID} / 7 + 1 ))
MODEL_IDX=$(( ${SLURM_ARRAY_TASK_ID} % 7 ))

# Map model index to config file
case ${MODEL_IDX} in
    0) CONFIG="configs/hierarchical.yaml" ;;
    1) CONFIG="configs/single_phylum.yaml" ;;
    2) CONFIG="configs/single_class.yaml" ;;
    3) CONFIG="configs/single_order.yaml" ;;
    4) CONFIG="configs/single_family.yaml" ;;
    5) CONFIG="configs/single_genus.yaml" ;;
    6) CONFIG="configs/single_species.yaml" ;;
esac

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Training fold ${FOLD} with config ${CONFIG}"
echo "Starting at: $(date)"

# Train the model
python scripts/train_fold.py --fold ${FOLD} --config ${CONFIG}

# If training succeeds, run evaluation
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting evaluation..."
    python scripts/evaluate_fold.py --fold ${FOLD} --config ${CONFIG} --mode best
    
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully."
    else
        echo "Evaluation failed for fold ${FOLD}, config ${CONFIG}"
        exit 1
    fi
else
    echo "Training failed for fold ${FOLD}, config ${CONFIG}"
    exit 1
fi

echo "Job completed at: $(date)"