"""
Classification heads for single-level and hierarchical classification
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SingleClassificationHead(nn.Module):
    """Single-level classification head"""
    
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, pooled_output):
        return self.classifier(pooled_output)


class HierarchicalClassificationHead(nn.Module):
    """Hierarchical classification head for taxonomic prediction"""
    
    def __init__(self, in_features: int, class_sizes: list, dropout_rate: float = 0.3):
        """
        Args:
            in_features: Dimensionality of the encoder's latent representation
            class_sizes: Number of classes at each hierarchical level 
                        [phylum, class, order, family, genus, species]
            dropout_rate: Dropout rate used in the classifier layers
        """
        super().__init__()
        
        self.in_features = in_features
        self.class_sizes = class_sizes
        self.num_levels = len(class_sizes)
        bottleneck_dim = 256
        self.classification_heads = nn.ModuleList()
        
        # Create classification layers for each level
        for i, class_size in enumerate(class_sizes):
            if i == 0:
                # First level uses only the pooled output
                self.classification_heads.append(
                    nn.Sequential(
                        nn.Linear(in_features, bottleneck_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(bottleneck_dim, class_size)
                    )
                )
                logger.info(f"Head {i} (phylum): input={in_features}, output={class_size}")
            else:
                # Subsequent levels use pooled output + previous level's predictions
                lvl_in_dim = in_features + class_sizes[i-1]
                lvl_out_dim = class_sizes[i]
                
                self.classification_heads.append(
                    nn.Sequential(
                        nn.Linear(lvl_in_dim, bottleneck_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(bottleneck_dim, lvl_out_dim)
                    )
                )
                logger.info(f"Head {i}: input={lvl_in_dim}, output={lvl_out_dim}")
    
    def forward(self, pooled_output):
        """
        Forward pass through hierarchical classification heads
        
        Args:
            pooled_output: Encoder output of shape (batch_size, in_features)
            
        Returns:
            List[torch.Tensor]: A list of classification logits for each hierarchical level
        """
        logits_list = []
        current_input = pooled_output
        
        for head in self.classification_heads:
            logits = head(current_input)  # Get predictions for current level
            logits_list.append(logits)
            # Concatenate original pooled output with current predictions for next level
            current_input = torch.cat((pooled_output, logits), dim=1)
        
        return logits_list  # List of tensors with shape (batch_size, num_classes) per level