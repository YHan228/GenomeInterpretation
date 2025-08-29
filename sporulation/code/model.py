import torch
import torch.nn as nn
import torch.nn.functional as F

class SporulationModel(nn.Module):
    def __init__(self):
        super(SporulationModel, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv1d(4, 32, kernel_size=200, padding=0, bias=True)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=100, stride=50)
        
        # Layer 2
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        
        # Layer 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=0, bias=True)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
    
        # Layer 4
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=0, bias=True)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.1)
    
        # Layer 5
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, padding=0, bias=True)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.1)
    
        # Layer 6
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=0, bias=True)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(0.1)
    
        # Layer 7
        self.conv7 = nn.Conv1d(128, 256, kernel_size=3, padding=0, bias=True)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout7 = nn.Dropout(0.1)

        # Final pooling and Dense layers
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 256, bias=True)
        self.fc2 = nn.Linear(256, 2, bias=True)

    def forward(self, x):
        # Input shape: (N, C, L), e.g. (batch, 4, 1000000)
        # PyTorch Conv1d expects (N, C, L), which is the default from the dataloader.

        # Conv layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        # Conv layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Conv layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Conv layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Conv layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        # Conv layer 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        # Conv layer 7
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        
        # Fully connected layers
        x = self.final_pool(x)
        x = x.squeeze(-1)
        x = self.fc_dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
