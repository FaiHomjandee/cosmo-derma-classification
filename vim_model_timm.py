# vim_model.py
import torch
import torch.nn as nn
import os
import timm

class ViM_Classifier(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        self.backbone = timm.create_model(
            'mambaout_femto.in1k',
            # ImageNet weights by default; set VIM_PRETRAINED=0 to train from scratch
            pretrained=os.environ.get('VIM_PRETRAINED', '1') != '0',
            num_classes=0
        )
        
        # ── FIX: probe the real output dim with a dummy pass ──
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_dim = self.backbone(dummy).shape[-1]
        print(f"[ViM] Detected feat_dim: {feat_dim}")

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, feat_dim)
        return self.head(features)