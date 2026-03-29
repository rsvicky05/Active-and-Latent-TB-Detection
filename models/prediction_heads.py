import torch
import torch.nn as nn

class PredictionHeads(nn.Module):
    def __init__(self, embed_dim=256):
        super(PredictionHeads, self).__init__()

        self.tb_classifier = nn.Linear(embed_dim, 2)
        self.active_latent_classifier = nn.Linear(embed_dim, 2)
        self.risk_regressor = nn.Linear(embed_dim, 1)

    def forward(self, fused_features):

        tb_output = self.tb_classifier(fused_features)
        active_latent_output = self.active_latent_classifier(fused_features)
        risk_output = self.risk_regressor(fused_features)

        return tb_output, active_latent_output, risk_output