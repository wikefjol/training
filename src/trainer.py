"""
Training logic for k-fold cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for fungal sequence classification"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict, 
                 output_dir: Path, fold: int):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Directory to save checkpoints
            fold: Fold number
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.fold = fold
        
        # Setup device
        self.device = torch.device(config['training']['device'])
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['training']['max_epochs']
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized for fold {fold}")
    
    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        """Create linear schedule with warmup"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, 
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip None batches (from filtering unknown labels)
            if batch is None:
                continue
                
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': total_correct / total_samples
                })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if batch is None:
                    continue
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': total_correct / total_samples,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'fold': self.fold,
            # Save label mappings
            'label_to_idx': self.train_loader.dataset.label_to_idx,
            'idx_to_label': self.train_loader.dataset.idx_to_label,
            # Save tokenizer vocab
            'tokenizer_vocab': self.train_loader.dataset.tokenizer.vocab if hasattr(self.train_loader.dataset.tokenizer, 'vocab') else None,
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
            logger.info(f"Saved best checkpoint with val_overall_accuracy: {metrics['val_overall_accuracy']:.4f}")
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training for fold {self.fold}")
        
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
                'train_overall_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_overall_accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['max_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Check for improvement
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
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