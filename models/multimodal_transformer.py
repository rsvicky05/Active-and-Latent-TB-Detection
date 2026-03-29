import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, embed_dim=256):
        super(MultimodalTransformer, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, img_feat, clin_feat):

        # Stack image and clinical features
        combined = torch.stack([img_feat, clin_feat], dim=1)

        attn_output, attn_weights = self.attention(combined, combined, combined)

        # Average fusion
        fused = attn_output.mean(dim=1)

        return fused, attn_weights