"""
MLM Trainer for pretraining BERT-like models
Adapted from old mlm_trainer.py to work with current codebase
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# Enable cuDNN autotuning
cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MLMTrainer:
    """Trainer for Masked Language Modeling pretraining"""
    
    def __init__(self, model, train_loader, val_loader, config: dict, output_dir: str,
                 accumulation_steps: int = 1, debug: bool = False):
        """
        Args:
            model: Model with MLM head
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Configuration dictionary
            output_dir: Directory to save outputs
            accumulation_steps: Gradient accumulation steps
            debug: Enable debug mode for detailed logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.accumulation_steps = accumulation_steps
        self.debug = debug
        
        # Training configuration
        training_config = config['training']
        self.max_epochs = training_config['max_epochs']
        self.patience = training_config.get('patience', 10)
        self.min_delta = training_config.get('min_delta', 1e-4)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.max_epochs // accumulation_steps
        warmup_steps = training_config.get('warmup_steps', total_steps // 10)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.use_amp = training_config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0
        self.global_step = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        logger.info(f"MLM Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  Batch size: {training_config['batch_size']}")
        logger.info(f"  Learning rate: {training_config['learning_rate']}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        
    def train(self):
        """Main training loop"""
        logger.info("Starting MLM pretraining...")
        
        for epoch in range(self.max_epochs):
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            self.train_metrics.append(train_metrics)
            
            # Validate epoch
            val_metrics = self._validate_epoch(epoch)
            self.val_metrics.append(val_metrics)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.max_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for improvement
            current_val_loss = val_metrics['loss']
            if current_val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = current_val_loss
                self.no_improvement_count = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.no_improvement_count += 1
                logger.info(f"  No improvement for {self.no_improvement_count} epochs")
            
            # Save regular checkpoint
            self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.no_improvement_count >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final results
        self._save_results()
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_metrics[-1]['loss'],
            'final_val_loss': self.val_metrics[-1]['loss'],
            'epochs_completed': len(self.train_metrics),
            'training_time': str(datetime.now())
        }
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch {epoch+1}",
            leave=False, 
            mininterval=5
        )
        
        # Time-based logging
        start_time = time.time()
        last_log_time = start_time
        log_interval_seconds = 900  # 15 minutes
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask, labels)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(input_ids, attention_mask, labels)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
            
            # Gradient accumulation
            if (i + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Compute metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            
            # Only count non-ignored positions
            mask_positions = (labels != -100)
            if mask_positions.any():
                correct += (predictions == labels)[mask_positions].sum().item()
                total += mask_positions.sum().item()
            
            # Time-based logging
            current_time = time.time()
            if current_time - last_log_time >= log_interval_seconds:
                batches_processed = i + 1
                percent_complete = (batches_processed / len(self.train_loader)) * 100
                current_loss = total_loss / batches_processed
                current_accuracy = correct / total if total > 0 else 0.0
                elapsed_minutes = (current_time - start_time) / 60
                
                # Log progress
                progress_msg = (
                    f"[TRAIN_PROGRESS] epoch={epoch+1} batch={batches_processed}/{len(self.train_loader)} "
                    f"pct={percent_complete:.1f}% loss={current_loss:.4f} acc={current_accuracy:.4f} "
                    f"elapsed={elapsed_minutes:.1f}m lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
                logger.info(progress_msg)
                last_log_time = current_time
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def _validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch+1}",
            leave=False,
            mininterval=5
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                
                logits = self.model(input_ids, attention_mask, labels)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                
                # Only count non-ignored positions
                mask_positions = (labels != -100)
                if mask_positions.any():
                    correct += (predictions == labels)[mask_positions].sum().item()
                    total += mask_positions.sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_pretrained_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def _save_results(self):
        """Save training results and metrics"""
        results = {
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'epochs_completed': len(self.train_metrics),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")