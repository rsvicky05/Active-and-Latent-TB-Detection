import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.clinical_encoder import ClinicalEncoder
from models.multimodal_transformer import MultimodalTransformer
from models.prediction_heads import PredictionHeads

class TBMultimodalModel(nn.Module):
    def __init__(self):
        super(TBMultimodalModel, self).__init__()

        self.image_encoder = ImageEncoder()
        self.clinical_encoder = ClinicalEncoder()

        self.fusion = MultimodalTransformer()
        self.heads = PredictionHeads()

    def forward(self, image, clinical):

        img_feat = self.image_encoder(image)
        clin_feat = self.clinical_encoder(clinical)

        fused, attn_weights = self.fusion(img_feat, clin_feat)

        tb_out, active_latent_out, risk_out = self.heads(fused)

        return tb_out, active_latent_out, risk_out, attn_weights