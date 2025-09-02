"""
Hierarchical trainer for taxonomic classification
Based on paper_dump implementation with k-fold support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class HierarchicalTrainer:
    """Trainer for hierarchical classification models"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 config: Dict,
                 output_dir: Path,
                 fold: int,
                 taxonomic_levels: List[str],
                 label_encoders: Dict = None,
                 vocab = None,
                 l1_lambda: float = 1e-4,
                 use_uncertainty_weighting: bool = False):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Directory to save checkpoints
            fold: Fold number
            taxonomic_levels: List of taxonomic levels
            label_encoders: Label encoders for each level
            vocab: Tokenizer vocabulary
            l1_lambda: L1 regularization strength
            use_uncertainty_weighting: Whether to use Kendall uncertainty weighting
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.fold = fold
        self.taxonomic_levels = taxonomic_levels
        self.num_levels = len(taxonomic_levels)
        self.label_encoders = label_encoders
        self.vocab = vocab
        self.l1_lambda = l1_lambda
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Device
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # Create abbreviated level names for logging
        self.level_abbr = self._create_level_abbreviations()
        
        # Loss function - using CrossEntropy for now (can switch to Entmax15)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.training_history = []
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Per-level metrics
        self.train_losses = {lvl: [] for lvl in taxonomic_levels}
        self.val_losses = {lvl: [] for lvl in taxonomic_levels}
        self.train_accs = {lvl: [] for lvl in taxonomic_levels}
        self.val_accs = {lvl: [] for lvl in taxonomic_levels}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_level_abbreviations(self) -> Dict[str, str]:
        """Create abbreviated names for taxonomic levels"""
        abbr = {}
        used = set()
        for level in self.taxonomic_levels:
            short = level[:3].lower()
            if short in used:
                short = level[:2].lower() + level[-1].lower()
            used.add(short)
            abbr[level] = short
        return abbr
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_steps = self.config['training'].get('warmup_steps', 500)
        total_steps = len(self.train_loader) * self.config['training']['max_epochs']
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=total_steps - warmup_steps
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    
    def _compute_hierarchical_loss(self, 
                                  logits_list: List[torch.Tensor],
                                  batch: Dict,
                                  log_vars: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute hierarchical loss with optional uncertainty weighting
        
        Args:
            logits_list: List of logits for each level
            batch: Batch dictionary with hierarchical labels
            log_vars: Optional log variance parameters for uncertainty weighting
            
        Returns:
            Total loss and per-level losses
        """
        # Compute per-level losses
        level_losses = []
        level_accs = []
        
        for i, level in enumerate(self.taxonomic_levels):
            # Get labels for this level from batch
            # The batch['output'] is a dict where each level contains dicts with 'encoded_label'
            # When batched by DataLoader, this becomes a list of encoded labels
            if isinstance(batch['output'][level]['encoded_label'], list):
                labels = torch.tensor(batch['output'][level]['encoded_label'], dtype=torch.long).to(self.device)
            else:
                labels = batch['output'][level]['encoded_label'].to(self.device)
            
            # Compute loss
            loss = self.criterion(logits_list[i], labels)
            level_losses.append(loss)
            
            # Compute accuracy
            with torch.no_grad():
                preds = torch.argmax(logits_list[i], dim=-1)
                acc = (preds == labels).float().mean()
                level_accs.append(acc.item())
        
        # Stack losses
        loss_tensor = torch.stack(level_losses)
        
        # Apply L1 regularization if specified
        if self.l1_lambda > 0:
            l1_losses = torch.stack([
                logits.abs().mean() * self.l1_lambda
                for logits in logits_list
            ])
            loss_tensor = loss_tensor + l1_losses
        
        # Combine losses
        if log_vars is not None and self.use_uncertainty_weighting:
            # Kendall uncertainty weighting: L_i / (2 * σ_i²) + log(σ_i)
            log_vars = torch.clamp(log_vars, min=-3.0, max=3.0)
            precision = torch.exp(-log_vars)
            weighted_losses = precision * loss_tensor + 0.5 * log_vars
            total_loss = weighted_losses.sum()
        else:
            # Simple sum
            total_loss = loss_tensor.sum()
        
        return total_loss, {
            'level_losses': [l.item() for l in level_losses],
            'level_accs': level_accs,
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        level_losses = {lvl: 0 for lvl in self.taxonomic_levels}
        level_correct = {lvl: 0 for lvl in self.taxonomic_levels}
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get log_vars if model has them
            log_vars = None
            if hasattr(self.model, 'log_vars'):
                log_vars = self.model.log_vars
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Forward pass
                logits_list = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss, metrics = self._compute_hierarchical_loss(
                    logits_list, batch, log_vars
                )
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for i, level in enumerate(self.taxonomic_levels):
                level_losses[level] += metrics['level_losses'][i] * batch_size
                level_correct[level] += metrics['level_accs'][i] * batch_size
            
            # Update progress bar
            postfix = {'loss': f"{loss.item():.4f}"}
            for level in self.taxonomic_levels:
                postfix[self.level_abbr[level]] = f"{metrics['level_accs'][self.taxonomic_levels.index(level)]:.3f}"
            pbar.set_postfix(postfix)
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'level_losses': {lvl: level_losses[lvl] / total_samples for lvl in self.taxonomic_levels},
            'level_accs': {lvl: level_correct[lvl] / total_samples for lvl in self.taxonomic_levels},
            'mean_acc': np.mean([level_correct[lvl] / total_samples for lvl in self.taxonomic_levels])
        }
        
        return epoch_metrics
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        level_losses = {lvl: 0 for lvl in self.taxonomic_levels}
        level_correct = {lvl: 0 for lvl in self.taxonomic_levels}
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get log_vars if model has them
                log_vars = None
                if hasattr(self.model, 'log_vars'):
                    log_vars = self.model.log_vars
                
                # Forward pass
                logits_list = self.model(input_ids, attention_mask)
                
                # Compute loss
                loss, metrics = self._compute_hierarchical_loss(
                    logits_list, batch, log_vars
                )
                
                # Update metrics
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                for i, level in enumerate(self.taxonomic_levels):
                    level_losses[level] += metrics['level_losses'][i] * batch_size
                    level_correct[level] += metrics['level_accs'][i] * batch_size
        
        # Calculate validation metrics
        val_metrics = {
            'loss': total_loss / total_samples,
            'level_losses': {lvl: level_losses[lvl] / total_samples for lvl in self.taxonomic_levels},
            'level_accs': {lvl: level_correct[lvl] / total_samples for lvl in self.taxonomic_levels},
            'mean_acc': np.mean([level_correct[lvl] / total_samples for lvl in self.taxonomic_levels])
        }
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'fold': self.fold,
            'taxonomic_levels': self.taxonomic_levels,
            # Save label encoders and vocab
            'label_encoders': self.label_encoders,
            'vocab': self.vocab,
            # Save tokenizer config
            'tokenizer_config': {
                'type': self.config['preprocessing']['tokenizer'],
                'kmer_size': self.config['preprocessing'].get('kmer_size'),
                'stride': self.config['preprocessing'].get('stride')
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with mean_acc: {metrics['mean_acc']:.4f}")
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info(f"Starting hierarchical training for fold {self.fold}")
        logger.info(f"Training on {self.num_levels} taxonomic levels: {self.taxonomic_levels}")
        
        for epoch in range(self.config['training']['max_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_mean_acc': train_metrics['mean_acc'],
                'val_loss': val_metrics['loss'],
                'val_mean_acc': val_metrics['mean_acc'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            # Add per-level metrics
            for level in self.taxonomic_levels:
                epoch_metrics[f'train_{level}_acc'] = train_metrics['level_accs'][level]
                epoch_metrics[f'val_{level}_acc'] = val_metrics['level_accs'][level]
            
            self.training_history.append(epoch_metrics)
            
            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['max_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['mean_acc']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['mean_acc']:.4f}"
            )
            
            # Log per-level accuracies
            for level in self.taxonomic_levels:
                logger.info(
                    f"  {level}: Train={train_metrics['level_accs'][level]:.4f}, "
                    f"Val={val_metrics['level_accs'][level]:.4f}"
                )
            
            # Check for improvement
            is_best = val_metrics['mean_acc'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['mean_acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if self.config['training']['save_best_only']:
                if is_best:
                    self.save_checkpoint(epoch_metrics, is_best=True)
            else:
                if epoch % self.config['training']['save_frequency'] == 0:
                    self.save_checkpoint(epoch_metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Return final metrics
        return {
            'fold': self.fold,
            'best_val_accuracy': self.best_val_accuracy,
            'final_epoch': self.current_epoch,
            'training_history': self.training_history
        }