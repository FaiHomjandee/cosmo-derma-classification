"""Official Vision Mamba (Vim-Tiny, hustvl/Vim) as a 7-class classifier for the CoSMO pipeline.
Env vars: VIM_REAL_PRETRAINED=1 (default) loads ImageNet-1k weights; 0 = random init."""
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Vim" / "vim"))
from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as _vim_tiny


def build_vim_real(num_classes=7):
    model = _vim_tiny(pretrained=False, num_classes=1000)
    if os.environ.get("VIM_REAL_PRETRAINED", "1") != "0":
        ck_files = glob.glob(str(HERE / "weights" / "**" / "*.pth"), recursive=True)
        assert ck_files, f"no Vim weights under {HERE / 'weights'} - run install_vim_env.sh"
        ck = torch.load(ck_files[0], map_location="cpu")
        missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
        print(f"[Vim] loaded {ck_files[0]} | missing {len(missing)} | unexpected {len(unexpected)}")
    else:
        print("[Vim] random initialisation")
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
