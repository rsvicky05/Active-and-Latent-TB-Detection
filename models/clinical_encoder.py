import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256):
        super(ClinicalEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.model(x)