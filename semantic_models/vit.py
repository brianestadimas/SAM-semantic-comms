import torch
import torch.nn as nn
from vit_pytorch import ViT

class SemanticEncoderVIT(nn.Module):
    def __init__(self):
        super(SemanticEncoderVIT, self).__init__()
        self.encoder = ViT(
            image_size=32,
            patch_size=4,
            num_classes=512,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=3)
        print(x.shape)
        encoded_features = self.encoder(x)  # Encode image features using ViT
        return encoded_features

class SemanticDecoderVIT(nn.Module):
    def __init__(self):
        super(SemanticDecoderVIT, self).__init__()
        self.encoder = ViT(
            image_size=8,
            patch_size=4,
            num_classes=512,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        encoded_features = self.encoder(x)  # Encode image features using ViT
        return encoded_features