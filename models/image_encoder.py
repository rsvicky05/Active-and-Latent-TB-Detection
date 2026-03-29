import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()

        # Step 1: Load trained ResNet18
        full_model = models.resnet18(pretrained=False)

        # Replace classifier to match training (2 classes)
        full_model.fc = nn.Linear(full_model.fc.in_features, 2)

        # Load trained CNN weights
        full_model.load_state_dict(
            torch.load("cnn_model.pth", map_location="cpu")
        )

        # Step 2: Remove classification layer (keep feature extractor)
        self.backbone = nn.Sequential(*list(full_model.children())[:-1])

        # ❌ FREEZE REMOVED
        # We DO NOT freeze backbone because GradCAM needs gradients

        # Step 3: Projection layer (for multimodal fusion)
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        # Extract CNN features
        features = self.backbone(x)          # (B, 512, 1, 1)

        features = features.view(features.size(0), -1)  # (B, 512)

        embedding = self.fc(features)        # (B, embed_dim)

        return embedding