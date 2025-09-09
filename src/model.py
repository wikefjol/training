"""
Simple BERT-like model for fungal sequence classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Dict, List, Union
import logging
try:
    from .heads import SingleClassificationHead, HierarchicalClassificationHead
except ImportError:
    from heads import SingleClassificationHead, HierarchicalClassificationHead

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 attention_probs_dropout_prob: float = 0.1):
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by "
                f"num_attention_heads {num_attention_heads}"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Convert attention mask to attention scores mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int,
                 intermediate_size: int, hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        
        return layer_output


class SequenceClassificationModel(nn.Module):
    """BERT-like model for sequence classification"""
    
    def __init__(self, vocab_size: int, 
                 num_classes: Union[int, List[int]],
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 hierarchical: bool = False,
                 hierarchical_dropout: float = 0.3):
        super().__init__()
        
        self.hierarchical = hierarchical
        
        self.config = {
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'hidden_dropout_prob': hidden_dropout_prob,
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'max_position_embeddings': max_position_embeddings,
            'hierarchical': hierarchical
        }
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.embedding_dropout = nn.Dropout(hidden_dropout_prob)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, intermediate_size,
                hidden_dropout_prob, attention_probs_dropout_prob
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Pooler
        self.pooler = nn.Linear(hidden_size, hidden_size)
        
        # Classification head
        if hierarchical:
            if not isinstance(num_classes, list):
                raise ValueError("For hierarchical classification, num_classes must be a list")
            self.classifier = HierarchicalClassificationHead(
                hidden_size, num_classes, hierarchical_dropout
            )
        else:
            if isinstance(num_classes, list):
                num_classes = num_classes[-1]  # Use species-level for single classification
            self.classifier = SingleClassificationHead(hidden_size, num_classes)
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Model created with {self.count_parameters():,} parameters")
    
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def load_pretrained_backbone(self, pretrained_path: str):
        """
        Load pretrained backbone weights (embeddings and transformer blocks)
        while keeping randomly initialized classification heads.
        
        Args:
            pretrained_path: Path to pretrained model checkpoint (.pt file)
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
        
        logger.info(f"Loading pretrained backbone from: {pretrained_path}")
        
        try:
            # Load pretrained checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['state_dict']
            else:
                pretrained_state_dict = checkpoint
            
            # Get current model state dict
            current_state_dict = self.state_dict()
            
            # Track what gets loaded
            loaded_keys = []
            skipped_keys = []
            
            # Load matching backbone weights (exclude classification heads)
            for key, value in pretrained_state_dict.items():
                # Skip classification head weights (these should be randomly initialized)
                if key.startswith('classifier.') or key.startswith('head.'):
                    skipped_keys.append(key)
                    continue
                
                # Load if key exists and shapes match
                if key in current_state_dict:
                    if current_state_dict[key].shape == value.shape:
                        current_state_dict[key] = value
                        loaded_keys.append(key)
                    else:
                        logger.warning(f"Shape mismatch for {key}: "
                                     f"current={current_state_dict[key].shape}, "
                                     f"pretrained={value.shape}")
                        skipped_keys.append(key)
                else:
                    skipped_keys.append(key)
            
            # Load the updated state dict
            self.load_state_dict(current_state_dict)
            
            logger.info(f"✅ Loaded pretrained backbone: {len(loaded_keys)} layers loaded")
            logger.info(f"📋 Loaded components: {', '.join(loaded_keys[:5])}{'...' if len(loaded_keys) > 5 else ''}")
            logger.info(f"🎯 Classification heads: randomly initialized (as expected)")
            
            if skipped_keys:
                logger.info(f"⏭️  Skipped {len(skipped_keys)} keys (classification heads or mismatched shapes)")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {e}")
        
        return self
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: Token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            For single classification: Logits [batch_size, num_classes]
            For hierarchical: List of logits for each level
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        # Pool the first token ([CLS] token)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.pooler(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def create_model(vocab_size: int, num_classes: Union[int, List[int]], config: Dict):
    """
    Create model from config
    
    Args:
        vocab_size: Vocabulary size
        num_classes: Number of output classes (int for single, list for hierarchical)
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    
    # Determine if hierarchical
    is_hierarchical = model_config.get('classification_type', 'single') == 'hierarchical'
    
    model = SequenceClassificationModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        hidden_dropout_prob=model_config['hidden_dropout_prob'],
        attention_probs_dropout_prob=model_config['attention_probs_dropout_prob'],
        max_position_embeddings=model_config['max_position_embeddings'],
        hierarchical=is_hierarchical,
        hierarchical_dropout=model_config.get('hierarchical_dropout', 0.3)
    )
    
    # Load pretrained backbone weights - REQUIRED for all training
    pretrained_path = config.get('pretrained_model_path')
    if not pretrained_path:
        raise ValueError(
            "No pretrained_model_path specified in config. "
            "Training from scratch is not allowed to prevent wasting compute resources. "
            "Please add pretrained_model_path to your config file."
        )
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}\n"
            "Training cannot continue without pretrained weights. "
            "Please verify the file exists and the path is correct."
        )
    
    logger.info(f"Loading pretrained backbone from: {pretrained_path}")
    model.load_pretrained_backbone(pretrained_path)
    
    # Add uncertainty weighting parameters if specified
    if is_hierarchical and model_config.get('use_uncertainty_weighting', False):
        if isinstance(num_classes, list):
            num_levels = len(num_classes)
        else:
            num_levels = len(model_config.get('taxonomic_levels', ['species']))
        model.log_vars = nn.Parameter(torch.zeros(num_levels))
    
    return model