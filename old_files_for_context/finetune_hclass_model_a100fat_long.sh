#!/bin/env bash
# Usage: sbatch finetune_model.sh <config_file> <hours>
# Example: sbatch finetune_model.sh configs/finetuning_5mer_config.json 12

#SBATCH -A NAISS2024-22-976
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 0-16:00:00
#SBATCH -J ft_model
#SBATCH -o /cephyr/users/filbern/Alvis/workspace/granular_control/sbatch_logs/finetune_hclass/%x_%j.out

CONFIG_FILE="$1"

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

source /mimer/NOBACKUP/groups/snic2022-22-552/filbern/environment/venv/bin/activate
cd /cephyr/users/filbern/Alvis/workspace/granular_control

python finetune_hclass_model.py --config "$CONFIG_FILE"
