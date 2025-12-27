import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnLocalDeep(nn.Module):
    """
    Repo version of CNN-dist (mislabeled). Actually a deeper CNN-local since first kernel is k=19.
    Kernel schedule: 19/7/7/3, Pool schedule: 4/4/3
    """
    def __init__(self, activation='relu', last_layer_mode='flat'):
        super(CnnLocalDeep, self).__init__()
        self.last_layer_mode = last_layer_mode
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'exponential':
            self.activation = torch.exp
        else:
            raise ValueError("Unsupported activation function")

        self.conv1 = nn.Conv1d(4, 24, kernel_size=19, padding=9, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(24, 32, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(32, 48, kernel_size=7, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(48)
        self.dropout3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, padding=0, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.4)
        
        if self.last_layer_mode == 'flat':
            self.flatten = nn.Flatten()
            fc1_in_features = 192  # 64 channels * 3 seq length
        elif self.last_layer_mode == 'noflat':
            self.pool = nn.AdaptiveMaxPool1d(1)
            fc1_in_features = 64  # 64 channels
        else:
            raise ValueError("Unsupported last_layer_mode")

        self.fc1 = nn.Linear(fc1_in_features, 96, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(96, 1, bias=True)  # Output layer keeps bias

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool1d(x, kernel_size=4)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = F.max_pool1d(x, kernel_size=4)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = F.max_pool1d(x, kernel_size=3, stride=3, padding=1)
        
        if self.last_layer_mode == 'flat':
            x = self.flatten(x)
        elif self.last_layer_mode == 'noflat':
            x = self.pool(x)
            x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

class CnnDist(nn.Module):
    """
    Paper version of CNN-dist. First kernel k=7 captures distributed motifs.
    Kernel schedule: 7/9/6/4, Pool schedule: 3/4/3
    """
    def __init__(self, activation='relu', last_layer_mode='flat'):
        super(CnnDist, self).__init__()
        self.last_layer_mode = last_layer_mode
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'exponential':
            self.activation = torch.exp
        else:
            raise ValueError("Unsupported activation function")

        # Paper: conv(24, k=7) -> conv(32, k=9)+pool(3) -> conv(48, k=6)+pool(4) -> conv(64, k=4)+pool(3)
        self.conv1 = nn.Conv1d(4, 24, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(24, 32, kernel_size=9, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(32, 48, kernel_size=6, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(48)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv1d(48, 64, kernel_size=4, padding=0, bias=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.4)

        if self.last_layer_mode == 'flat':
            self.flatten = nn.Flatten()
            # Calculate output size: 200 -> conv1(k7,p3) -> 200 -> conv2(k9,p4)+pool3 -> 66
            # -> conv3(k6,p0)+pool4 -> 15 -> conv4(k4,p0)+pool3 -> 4
            # 64 channels * 4 = 256
            fc1_in_features = 256
        elif self.last_layer_mode == 'noflat':
            self.pool = nn.AdaptiveMaxPool1d(1)
            fc1_in_features = 64
        else:
            raise ValueError("Unsupported last_layer_mode")

        self.fc1 = nn.Linear(fc1_in_features, 96, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(96, 1, bias=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool1d(x, kernel_size=3)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = F.max_pool1d(x, kernel_size=4)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = F.max_pool1d(x, kernel_size=3)

        if self.last_layer_mode == 'flat':
            x = self.flatten(x)
        elif self.last_layer_mode == 'noflat':
            x = self.pool(x)
            x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class CnnLocal(nn.Module):
    def __init__(self, activation='relu', last_layer_mode='flat'):
        super(CnnLocal, self).__init__()
        self.last_layer_mode = last_layer_mode
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'exponential':
            self.activation = torch.exp
        else:
            raise ValueError("Unsupported activation function")

        self.conv1 = nn.Conv1d(4, 24, kernel_size=19, padding=9, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(24, 48, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(48)
        self.dropout2 = nn.Dropout(0.2)
        
        if self.last_layer_mode == 'flat':
            self.flatten = nn.Flatten()
            fc1_in_features = 48 * 2  # 48 channels * 2 seq length
        elif self.last_layer_mode == 'noflat':
            self.pool = nn.AdaptiveMaxPool1d(1)
            fc1_in_features = 48  # 48 channels
        else:
            raise ValueError("Unsupported last_layer_mode")

        self.fc1 = nn.Linear(fc1_in_features, 96, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(96)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(96, 1, bias=True)  # Output layer keeps bias

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = F.max_pool1d(x, kernel_size=50)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        if self.last_layer_mode == 'flat':
            x = self.flatten(x)
        elif self.last_layer_mode == 'noflat':
            x = self.pool(x)
            x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x 