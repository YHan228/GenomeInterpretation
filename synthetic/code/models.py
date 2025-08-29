"""
Neural network model architectures for synthetic sequence experiments.
Includes models from both toy_slurm.py and merged_experiment.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# --------------------------------------------------------------------------- #
# Sanity Check Models
# --------------------------------------------------------------------------- #

class LogisticRegression(nn.Module):
    """Logistic regression on k-mer counts as sanity check"""
    def __init__(self, k=6):
        super().__init__()
        self.k = k
        self.n_features = 4 ** k  # Number of possible k-mers
        self.fc = nn.Linear(self.n_features, 1)
        
    def extract_kmer_counts(self, x: torch.Tensor) -> torch.Tensor:
        """Extract k-mer counts from one-hot encoded sequences (vectorized)"""
        batch_size, _, seq_len = x.shape
        
        # Convert one-hot to indices
        seq_indices = torch.argmax(x, dim=1)  # (batch_size, seq_len)

        # Get sliding windows of size k
        # Shape: (batch_size, seq_len - k + 1, k)
        kmers = seq_indices.unfold(dimension=1, size=self.k, step=1)

        # Create powers of 4 for base conversion (view as a base-4 number)
        # Shape: (k,)
        powers = 4 ** torch.arange(self.k - 1, -1, -1, device=x.device, dtype=torch.long)
        
        # Convert k-mer windows to single integer indices
        # (batch_size, seq_len - k + 1, k) * (k,) -> sum -> (batch_size, seq_len - k + 1)
        kmer_indices = (kmers.long() * powers).sum(dim=2)

        # Count occurrences of each k-mer index for each sequence in the batch
        counts = torch.zeros(batch_size, self.n_features, device=x.device, dtype=torch.float32)
        
        # Use scatter_add_ for efficient, batched counting
        ones = torch.ones_like(kmer_indices, dtype=torch.float32)
        counts.scatter_add_(dim=1, index=kmer_indices, src=ones)
                
        # Normalize by number of k-mers
        n_kmers = seq_len - self.k + 1
        if n_kmers > 0:
            counts = counts / n_kmers
            
        return counts
    
    def forward(self, x):
        features = self.extract_kmer_counts(x)
        logits = self.fc(features)
        return logits.squeeze(-1), features


# --------------------------------------------------------------------------- #
# Legacy Model (backwards compatibility)
# --------------------------------------------------------------------------- #

class TinyCNNv0(nn.Module):
    """Original simple architecture for backwards compatibility"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 13, padding=6)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits.squeeze(-1), x


# --------------------------------------------------------------------------- #
# Vanilla Model (for toy_slurm experiments)
# --------------------------------------------------------------------------- #

class TinyCNN_Vanilla(nn.Module):
    """
    Simpler architecture for vanilla experiments (toy_slurm.py).
    Features:
    - 3 convolutional layers (32, 64, 128 channels)
    - Exponential activation after first layer
    - Localist pooling (max pooling with window size 50)
    - No dilation
    - Higher dropout (0.5) in FC layer
    """
    def __init__(self):
        super().__init__()
        # User-specified k1, with sensible defaults for subsequent layers
        self.k1, self.k2, self.k3 = 30, 3, 3

        # Calculate padding to keep sequence length constant *before* pooling
        p1 = (self.k1 - 1) // 2
        # Padding for subsequent layers operating on pooled output
        p2 = (self.k2 - 1) // 2
        p3 = (self.k3 - 1) // 2

        # Conv Block 1
        self.conv1 = nn.Conv1d(4, 32, kernel_size=self.k1, padding=p1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)

        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.k2, padding=p2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)

        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.k3, padding=p3)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.1)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Conv Block 1: Motif scanning
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x)
        x = self.dropout1(x)
        
        # Localist pooling: Drastically downsample to get motif presence features
        x = F.max_pool1d(x, 50)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Conv Block 3
        conv3_out = self.conv3(x)
        x = self.bn3(conv3_out)
        x = F.relu(x)
        x = self.dropout3(x)

        # FC Layer
        x = self.pool(x).squeeze(-1)
        x = self.fc_dropout(x)
        logits = self.fc(x)
        return logits.squeeze(-1), conv3_out
    
    def receptive_field(self) -> int:
        """Return the receptive field size of the first convolutional layer"""
        return self.k1


# --------------------------------------------------------------------------- #
# Complex Model (for merged_experiment.py)
# --------------------------------------------------------------------------- #

class TinyCNN(nn.Module):
    """
    Complex architecture for merged experiments.
    Used in merged_experiment.py.
    
    Features:
    - 4 convolutional layers (64, 128, 256, 512 channels)
    - Exponential activation after first layer
    - Localist pooling (max pooling with window size 50)
    - Dilated convolutions in deeper layers for increased receptive field
    - Batch normalization with specific settings (eps=1e-5, momentum=0.05)
    - Lower dropout (0.2) in FC layer
    - AdaptiveAvgPool1d instead of AdaptiveMaxPool1d
    """
    def __init__(self):
        super().__init__()
        # User-specified kernel sizes
        self.k1, self.k2 = 30, 3

        # Calculate padding to keep sequence length constant
        p1 = (self.k1 - 1) // 2
        p2 = (self.k2 - 1) // 2

        # Conv Block 1: Motif scanning layer
        self.conv1 = nn.Conv1d(4, 64, kernel_size=self.k1, padding=p1)
        self.bn1 = nn.BatchNorm1d(64, eps=1e-5, momentum=0.05)
        self.dropout1 = nn.Dropout(0.1)

        # Conv Block 2 - dilation=3, effective RF after pool: 3*3*50=450bp
        self.conv2 = nn.Conv1d(64, 128, kernel_size=self.k2, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.05)
        self.dropout2 = nn.Dropout(0.1)

        # Conv Block 3 - dilation=5, effective RF: 3*5*50=750bp
        self.conv3 = nn.Conv1d(128, 256, kernel_size=self.k2, padding=5, dilation=5)
        self.bn3 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.05)
        self.dropout3 = nn.Dropout(0.1)

        # Conv Block 4 - dilation=7, effective RF: 3*7*50=1050bp
        self.conv4 = nn.Conv1d(256, 512, kernel_size=self.k2, padding=7, dilation=7)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.05)
        self.dropout4 = nn.Dropout(0.1)

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_dropout = nn.Dropout(0.2)  # Reduced from 0.5
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # Conv Block 1: Motif scanning
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x)  # Exponential activation for motif matching
        x = self.dropout1(x)
        
        # Localist pooling - reduces spatial resolution by 50x
        x = F.max_pool1d(x, 50)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Conv Block 4
        conv4_out = self.conv4(x)
        x = self.bn4(conv4_out)
        x = F.relu(x)
        x = self.dropout4(x)

        # FC Layer
        x = self.pool(x).squeeze(-1)
        x = self.fc_dropout(x)
        logits = self.fc(x)
        return logits.squeeze(-1), conv4_out
    
    def receptive_field(self) -> int:
        """Return the receptive field size of the first convolutional layer"""
        return self.k1


# --------------------------------------------------------------------------- #
# Model Selection Utility
# --------------------------------------------------------------------------- #

def get_model(model_name: str) -> nn.Module:
    """
    Factory function to get model by name.
    
    Available models:
    - 'tinycnn': Complex architecture with 4 layers and dilations (for merged experiments)
    - 'tinycnn_vanilla': Simpler architecture with 3 layers (for vanilla experiments) 
    - 'tinycnn_v0': Legacy version for backwards compatibility
    - 'logistic': Logistic regression on k-mer counts
    """
    models = {
        'tinycnn': TinyCNN,
        'tinycnn_vanilla': TinyCNN_Vanilla,
        'tinycnn_v0': TinyCNNv0,
        'logistic': LogisticRegression,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]() 