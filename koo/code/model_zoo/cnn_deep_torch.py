import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnDeepTorch(nn.Module):
    def __init__(self, input_shape=200):
        super().__init__()
        
        multiplier = 2 if input_shape == 1000 else 1

        # Layer 1
        self.conv1 = nn.Conv1d(4, 32 * multiplier, kernel_size=19, padding='same')
        self.bn1 = nn.BatchNorm1d(32 * multiplier)
        self.dropout1 = nn.Dropout(0.1)

        # Layer 2
        self.conv2 = nn.Conv1d(32 * multiplier, 48 * multiplier, kernel_size=7, padding='same')
        self.bn2 = nn.BatchNorm1d(48 * multiplier)
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        # Layer 3
        self.conv3 = nn.Conv1d(48 * multiplier, 96 * multiplier, kernel_size=7) # padding='valid'
        self.bn3 = nn.BatchNorm1d(96 * multiplier)
        self.dropout3 = nn.Dropout(0.3)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        # Layer 4
        self.conv4 = nn.Conv1d(96 * multiplier, 128 * multiplier, kernel_size=3) # padding='valid'
        self.bn4 = nn.BatchNorm1d(128 * multiplier)
        self.dropout4 = nn.Dropout(0.4)
        self.pool4 = nn.MaxPool1d(kernel_size=3)
        
        # Layer 5 - Dense
        # The input size to the dense layer depends on the output from conv4 and pool4
        # We need to calculate this dynamically.
        # For input_shape=200:
        # After conv1 ('same'): 200
        # After conv2 ('same'): 200 -> pool2(4): 50
        # After conv3 ('valid', k=7): 50 - 7 + 1 = 44 -> pool3(4): 11
        # After conv4 ('valid', k=3): 11 - 3 + 1 = 9 -> pool4(3): 3
        # Flattened size = 128 * multiplier * 3
        self.dense1 = nn.Linear(128 * multiplier * 3, 512 * multiplier)
        self.bn5 = nn.BatchNorm1d(512 * multiplier)
        self.dropout5 = nn.Dropout(0.5)

        # Output Layer
        self.fc_out = nn.Linear(512 * multiplier, 12)

    def forward(self, x):
        # x: (batch, 4, 200)
        
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x) # The original script uses a flexible activation, we default to exp as used before
        x = self.dropout1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.pool4(x)

        # Layer 5
        x = x.flatten(start_dim=1)
        x = self.dense1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        # Output
        logits = self.fc_out(x)
        
        # The training script will handle the sigmoid activation via BCEWithLogitsLoss
        # We also don't need to return the intermediate conv output for this model
        return logits 