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


class MLMHead(nn.Module):
    """Masked Language Modeling head for pretraining"""
    
    def __init__(self, in_features, hidden_layer_size, out_features, dropout_rate=0.1):
        """
        Args:
            in_features: Size of the input features (hidden_size)
            hidden_layer_size: Size of the hidden layer  
            out_features: Size of the output features (vocab_size)
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.in_features = in_features
        self.hidden_layer_size = hidden_layer_size
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        
        self.sequential = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_layer_size, self.out_features)
        )

    def forward(self, x):
        return self.sequential(x)


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
    """BERT-like model for sequence classification and pretraining"""
    
    def __init__(self, vocab_size: int, 
                 num_classes: Union[int, List[int]] = None,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 hierarchical: bool = False,
                 hierarchical_dropout: float = 0.3,
                 mode: str = "classify",
                 mlm_dropout_rate: float = 0.1):
        super().__init__()
        
        self.hierarchical = hierarchical
        self.mode = mode
        
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
            'hierarchical': hierarchical,
            'mode': mode
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
        
        # Pooler (only for classification mode)
        if mode == "classify":
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
        
        elif mode == "pretrain":
            # MLM head for pretraining
            self.mlm_head = MLMHead(
                in_features=hidden_size,
                hidden_layer_size=hidden_size, 
                out_features=vocab_size,
                dropout_rate=mlm_dropout_rate
            )
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'classify' or 'pretrain'")
        
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
    
    def _map_pretrained_key(self, pretrained_key: str) -> str:
        """
        Map HuggingFace BERT parameter names to our custom model names.
        
        Args:
            pretrained_key: Key from pretrained model (HuggingFace format)
            
        Returns:
            Corresponding key in our model, or None if not mappable
        """
        # HuggingFace -> Our model key mappings
        key_mappings = {
            # Embeddings
            'bert.embeddings.word_embeddings.weight': 'word_embeddings.weight',
            'bert.embeddings.position_embeddings.weight': 'position_embeddings.weight',
            'bert.embeddings.LayerNorm.weight': 'embedding_layer_norm.weight',
            'bert.embeddings.LayerNorm.bias': 'embedding_layer_norm.bias',
        }
        
        # Direct mapping first
        if pretrained_key in key_mappings:
            return key_mappings[pretrained_key]
        
        # Pattern-based mapping for transformer layers
        if pretrained_key.startswith('bert.encoder.layer.'):
            # Extract layer number and component path
            parts = pretrained_key.split('.')
            if len(parts) >= 6:
                layer_num = parts[3]  # bert.encoder.layer.{N}
                component_path = '.'.join(parts[4:])  # attention.self.key.weight, etc.
                
                # Map component paths
                component_mappings = {
                    # Self-attention
                    'attention.self.query.weight': 'attention.query.weight',
                    'attention.self.query.bias': 'attention.query.bias',
                    'attention.self.key.weight': 'attention.key.weight',
                    'attention.self.key.bias': 'attention.key.bias', 
                    'attention.self.value.weight': 'attention.value.weight',
                    'attention.self.value.bias': 'attention.value.bias',
                    'attention.output.dense.weight': 'attention_output.weight',
                    'attention.output.dense.bias': 'attention_output.bias',
                    'attention.output.LayerNorm.weight': 'attention_layer_norm.weight',
                    'attention.output.LayerNorm.bias': 'attention_layer_norm.bias',
                    # Feed-forward
                    'intermediate.dense.weight': 'intermediate.weight',
                    'intermediate.dense.bias': 'intermediate.bias',
                    'output.dense.weight': 'output.weight',
                    'output.dense.bias': 'output.bias',
                    'output.LayerNorm.weight': 'output_layer_norm.weight',
                    'output.LayerNorm.bias': 'output_layer_norm.bias',
                }
                
                if component_path in component_mappings:
                    return f'transformer_blocks.{layer_num}.{component_mappings[component_path]}'
        
        return None

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
            mapped_keys = []
            
            # Detect checkpoint format
            is_huggingface_format = any(key.startswith('bert.') for key in pretrained_state_dict.keys())
            
            # Load matching backbone weights (exclude classification heads)
            for pretrained_key, value in pretrained_state_dict.items():
                # Skip classification head weights (these should be randomly initialized)
                if (pretrained_key.startswith('classifier.') or 
                    pretrained_key.startswith('head.') or
                    pretrained_key.startswith('bert.pooler.') or
                    pretrained_key.startswith('mlm_head.')):  # Also skip MLM head
                    skipped_keys.append(pretrained_key)
                    continue
                
                # Determine target key based on checkpoint format
                if is_huggingface_format:
                    # Map HuggingFace key to our model key
                    mapped_key = self._map_pretrained_key(pretrained_key)
                    target_key = mapped_key
                else:
                    # Native format - use key directly (no mapping needed)
                    target_key = pretrained_key
                
                if target_key and target_key in current_state_dict:
                    if current_state_dict[target_key].shape == value.shape:
                        current_state_dict[target_key] = value
                        loaded_keys.append(target_key)
                        if is_huggingface_format:
                            mapped_keys.append(f"{pretrained_key} -> {target_key}")
                        else:
                            mapped_keys.append(pretrained_key)
                    else:
                        logger.warning(f"Shape mismatch for {pretrained_key} -> {target_key}: "
                                     f"current={current_state_dict[target_key].shape}, "
                                     f"pretrained={value.shape}")
                        skipped_keys.append(pretrained_key)
                else:
                    skipped_keys.append(pretrained_key)
            
            # Load the updated state dict
            self.load_state_dict(current_state_dict)
            
            logger.info(f"✅ Loaded pretrained backbone: {len(loaded_keys)} layers loaded")
            logger.info(f"📋 Loaded components: {', '.join(loaded_keys[:5])}{'...' if len(loaded_keys) > 5 else ''}")
            logger.info(f"🔗 Key mappings: {len(mapped_keys)} successful")
            logger.info(f"🎯 Classification heads: randomly initialized (as expected)")
            
            if skipped_keys:
                logger.info(f"⏭️  Skipped {len(skipped_keys)} keys (classification heads or unmappable)")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {e}")
        
        return self
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            input_ids: Token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length] 
            labels: For MLM mode, masked token labels [batch_size, seq_length]
            
        Returns:
            For single classification: Logits [batch_size, num_classes]
            For hierarchical: List of logits for each level
            For MLM: Token prediction logits [batch_size, seq_length, vocab_size]
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
        
        if self.mode == "pretrain":
            # MLM: predict all tokens
            mlm_logits = self.mlm_head(hidden_states)  # [batch_size, seq_length, vocab_size]
            return mlm_logits
            
        elif self.mode == "classify":
            # Classification: pool and classify
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
                    Can be None for pretraining mode
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_config = config['model']
    
    # Get mode (pretrain or classify)
    mode = model_config.get('mode', 'classify')
    
    # Determine if hierarchical (only relevant for classification)
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
        hierarchical_dropout=model_config.get('hierarchical_dropout', 0.3),
        mode=mode,
        mlm_dropout_rate=model_config.get('mlm_dropout_rate', 0.1)
    )
    
    # Handle pretrained weights loading
    if mode == "pretrain":
        # For pretraining, pretrained path is optional (can start from scratch or continue)
        pretrained_path = config.get('pretrained_model_path')
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained backbone from: {pretrained_path}")
            model.load_pretrained_backbone(pretrained_path)
        else:
            logger.info("No pretrained model specified for pretraining - training from scratch")
            
    elif mode == "classify":
        # For classification, resolve pretrained model path with fallback strategy
        from pathlib import Path
        
        # Strategy 1: Use explicit config path if provided
        explicit_path = config.get('pretrained_model_path')
        pretrained_path = None
        
        if explicit_path and os.path.exists(explicit_path):
            pretrained_path = explicit_path
            logger.info(f"Using explicit pretrained model path: {pretrained_path}")
        else:
            # Strategy 2: Auto-resolve from experiment structure
            experiments_dir = os.path.expandvars(os.getenv('EXPERIMENTS_DIR'))
            experiment_base = Path(experiments_dir) / config['experiment']['fold_type'] / \
                              config['experiment']['dataset_size']
            
            auto_pretrained_path = experiment_base / 'models' / config['experiment']['union_type'] / \
                                   'pretrained_model' / 'best_pretrained_model.pt'
            
            if auto_pretrained_path.exists():
                pretrained_path = str(auto_pretrained_path)
                logger.info(f"Auto-resolved pretrained model path: {pretrained_path}")
            else:
                # Strategy 3: Fallback to explicit config path even if it didn't exist initially
                if explicit_path:
                    logger.warning(f"Auto-resolve failed, attempting fallback to config path: {explicit_path}")
                    pretrained_path = explicit_path  # Will be verified below
                else:
                    raise FileNotFoundError(
                        f"No pretrained model found at:\n"
                        f"  Auto-resolved: {auto_pretrained_path}\n"
                        f"  Config path: {explicit_path or 'Not specified'}\n"
                        "Please run pretraining first or specify valid pretrained_model_path in config."
                    )
        
        # Verify final path exists
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model not found: {pretrained_path}\n"
                "Training cannot continue without pretrained weights. "
                "Please verify the file exists and run pretraining if needed."
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