# Training Module

Clean, independent k-fold cross-validation training module for fungal classification.

## Overview

This module implements a simple BERT-like transformer model for DNA sequence classification. It reads k-fold assignments from `experiment_prep` outputs and trains models independently for each fold.

### Key Features
- **Completely Independent**: No dependencies on paper_dump or other modules
- **Clean Implementation**: Simple, readable code with just what we need
- **K-fold Support**: Handles both sequence-level (exp1) and species-level (exp2) folds
- **Flexible Configuration**: Separate .env for paths, config.yaml for parameters

## Architecture

The module consists of:
- **Data Loading** (`src/data.py`): Dataset classes and k-fold splitting
- **Preprocessing** (`src/preprocessing.py`): K-mer and character tokenizers
- **Model** (`src/model.py`): BERT-like transformer architecture
- **Training** (`src/trainer.py`): Training loop with early stopping
- **Scripts** (`scripts/`): Entry points for training

## Setup

1. Clone the repository:
```bash
git clone https://github.com/wikefjol/training.git
cd training
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your machine-specific paths
```

3. Install dependencies:
```bash
pip install torch pandas pyyaml python-dotenv tqdm
```

## Configuration

### .env File
Contains machine-specific paths:
```bash
MACHINE=alvis  # or macbook
DATA_ROOT=/mimer/NOBACKUP/groups/snic2022-22-552/filbern
EXPERIMENTS_DIR=${DATA_ROOT}/fungal_classification/experiments
MODELS_DIR=${DATA_ROOT}/fungal_classification/models
```

### config.yaml
Contains training parameters:
```yaml
experiment:
  union_type: "standard"  # a+b+d (default) or "conservative" (a+b+c)
  fold_type: "exp1_sequence_fold"  # or "exp2_species_fold"
  dataset_size: "debug_5genera_10fold"  # or "full_10fold"

model:
  hidden_size: 256  # Small for debug
  num_hidden_layers: 4
  num_attention_heads: 4

training:
  batch_size: 32
  learning_rate: 5e-5
  max_epochs: 10
```

## Usage

### Train a Single Fold
```bash
python scripts/train_fold.py --fold 1
```

### Train All Folds
```bash
python scripts/train_all_folds.py --folds all
```

### Train Specific Folds
```bash
python scripts/train_all_folds.py --folds 1,2,3
```

### Use Custom Config
```bash
python scripts/train_fold.py --fold 1 --config configs/large_model.yaml
```

## Data Structure

Expects data from experiment_prep in the following structure:
```
$EXPERIMENTS_DIR/
├── exp1_sequence_fold/
│   ├── full_10fold/
│   │   ├── standard.csv      # a+b+d union
│   │   └── conservative.csv  # a+b+c union
│   └── debug_5genera_10fold/
│       ├── standard.csv
│       └── conservative.csv
└── exp2_species_fold/
    └── (same structure)
```

Each CSV contains:
- `sequence`: DNA sequence
- `genus`: Genus name
- `species`: Species name
- `fold_exp1` or `fold_exp2`: Fold assignment (1-10)

## Output Structure

Models are saved to:
```
$MODELS_DIR/
└── {union_type}_{fold_type}_{dataset_size}/
    ├── fold_1/
    │   ├── checkpoint_best.pt
    │   ├── training_history.json
    │   ├── config.json
    │   └── results.json
    ├── fold_2/
    │   └── ...
    └── kfold_summary.json
```

## Model Architecture

Simple BERT-like transformer:
- **Embedding**: Token + Position embeddings
- **Transformer Blocks**: Multi-head self-attention + Feed-forward
- **Classification Head**: Pooler + Linear classifier

Default sizes:
- **Debug**: 4 layers, 256 hidden, 4 heads (~2M parameters)
- **Small**: 6 layers, 512 hidden, 8 heads (~15M parameters)
- **Full**: 12 layers, 768 hidden, 12 heads (~85M parameters)

## Testing Strategy

1. **Debug Test** (10k sequences):
   - Use `debug_5genera_10fold` dataset
   - Small model (4 layers)
   - 1-2 epochs
   - Should take ~5 minutes per fold

2. **Full Test** (1.4M sequences):
   - Use `full_10fold` dataset
   - Full model (12 layers)
   - 10+ epochs
   - Will take several hours per fold

## Integration with Pipeline

This module is part of the modular fungal classification pipeline:

1. **BLAST** → Filters sequences (creates a, b, c, d datasets)
2. **Experiment Prep** → Creates unions (a+b+d, a+b+c) and assigns k-folds
3. **Training** (this module) → Trains models for each fold
4. **Evaluation** → Analyzes results across folds

## Notes

- Start with debug datasets for testing
- Monitor GPU memory usage with larger models
- Use mixed precision training for faster training on V100/A100
- Each fold trains independently - can be parallelized across GPUs