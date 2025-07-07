import torch
import torch.nn as nn
import numpy as np

# Fixed version of your PatchProducer class
class PatchProducer(nn.Module):
    def __init__(self, input_dim=22, patch_len=16, dropout=0.2, channels=3):  # Fixed __init__
        super(PatchProducer, self).__init__()
        self.channels = channels
        self.patch_len = patch_len
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, patch_len * patch_len * channels)
        self.batch_norm = nn.BatchNorm1d(patch_len * patch_len * channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Use torch.relu instead of nn.functional.relu
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.batch_norm(x)
        return x.reshape(x.shape[0], self.channels, self.patch_len, self.patch_len)