#!/usr/bin/env python
# coding: utf-8
"""
R2.2 evidence: are CoSMO's synthetic images class-consistent, not copies, and stable?

1. Label consistency: for each fold, the Aug-only classifier of that fold
   (trained on REAL images only) predicts every synthetic image; agreement with
   the intended class is reported per class, next to the same classifier's
   recall on REAL test images of that class as a reference.
2. Not copies: for a sample of synthetic images, the nearest REAL training image
   of the same class is found in ImageNet ResNet-18 feature space; cosine
   similarity and SSIM to that neighbour are compared with the same statistics
   for real images vs their nearest OTHER real image (real-real reference).
3. Stability: every number is computed per fold and summarised as mean +- SD.

Conditions: CoSMO (alpha 0.5) and AE+SMOTE (alpha 1.0). Each classifier is
checked on the synthetic data of the latent size it used in the paper.
Outputs: revision_analysis/r22_label_consistency.csv, r22_nearest_neighbour.csv

    python revision_analysis/r22_synthetic_checks.py                     # everything
    python revision_analysis/r22_synthetic_checks.py --folds 1 --models resnet18   # quick test
"""
import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

B = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"; OUT.mkdir(exist_ok=True)
NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
MODELS = ["vim", "resnet18", "efficientnet_b0", "mobilenet_v2", "vit"]
BEST_NZ = {"resnet18": 128, "vit": 128, "vim": 512, "efficientnet_b0": 512, "mobilenet_v2": 512}
SOURCES = {"cosmo": B / "outputs" / "balanced", "ae_smote": B / "outputs" / "balanced_ae_smote"}
ALPHA = {"cosmo": 0.5, "ae_smote": 1.0}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(name):
    from torchvision import models
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, 7)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    elif name == "vim":
        import timm

        class ViM_Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("mambaout_femto.in1k", pretrained=False, num_classes=0)
                with torch.no_grad():
                    fd = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
                self.head = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, 7))

            def forward(self, x):
                return self.head(self.backbone(x))
        m = ViM_Classifier()
    elif name == "vit":
        from vit_pytorch import ViT
        m = ViT(image_size=224, patch_size=32, num_classes=7, dim=128, depth=6, heads=8, mlp_dim=256, channels=3)
    return m


@torch.no_grad()
def predict(net, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        o = net(X[i:i + bs].to(DEV))
        out.append((o[0] if isinstance(o, tuple) else o).argmax(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def features(X, fx, bs=256):
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(X), bs):
        f = fx((X[i:i + bs].to(DEV).clamp(0, 1) - mean) / std)
        out.append(nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return structural_similarity(a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy(),
                                 channel_axis=2, data_range=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--nn-per-class", type=int, default=200, help="synthetic images per class for the copy check")
    ap.add_argument("--tag", default="", help="suffix for the output CSVs, e.g. part1, so parallel runs do not overwrite each other")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    from torchvision import models
    fx = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    fx.fc = nn.Identity()
    fx = fx.to(DEV).eval()

    lab_rows, nn_rows = [], []
    for fold in a.folds:
        train_loader_ae, _, test_loader = torch.load(B / "data" / "prepared" / f"dermamnist_fold{fold}_data.pth", weights_only=False)
        Xr, yr = train_loader_ae.dataset.tensors[0], train_loader_ae.dataset.tensors[1].view(-1).numpy()
        Xt, yt = test_loader.dataset.tensors[0], test_loader.dataset.tensors[1].view(-1).numpy()
        n_real = len(yr)
        del train_loader_ae, test_loader

        # reference recall of each Aug-only classifier on real test images
        nets, ref = {}, {}
        for m in a.models:
            net = build_model(m)
            net.load_state_dict(torch.load(B / "outputs" / "ablation" / "Results_nz0" / "Saved_Models" /
                                           f"imbalanced_data_{m}_lr0.0001_z0_fold{fold}.pth", map_location="cpu"))
            nets[m] = net.to(DEV).eval()
            pt = predict(nets[m], Xt)
            ref[m] = [float((pt[yt == c] == c).mean()) for c in range(7)]

        Fr = features(Xr, fx)                                  # real training features (copy check)
        for src in a.sources:
            for nz in sorted({BEST_NZ[m] for m in a.models}):
                path = SOURCES[src] / f"dermamnist_fold{fold}_lr0.0002_alpha{ALPHA[src]}_z{nz}.pth"
                if not path.exists():
                    print(f"[MISSING] {path}"); continue
                ck = torch.load(path, weights_only=False)
                Xs, ys = ck["balanced_data"][n_real:], ck["balanced_labels"].view(-1)[n_real:].numpy()
                assert torch.equal(ck["balanced_data"][:5], Xr[:5]), "real images are not first in the balanced file"
                del ck; gc.collect()
                print(f"fold {fold} | {src} nz{nz} | {len(ys)} synthetic images")

                # 1. label consistency
                for m in [m for m in a.models if BEST_NZ[m] == nz]:
                    ps = predict(nets[m], Xs)
                    for c in range(7):
                        mk = ys == c
                        if mk.any():
                            lab_rows.append(dict(fold=fold, source=src, n_z=nz, model=m, cls=NAMES[c], n_syn=int(mk.sum()),
                                                 agreement=round(float((ps[mk] == c).mean()), 4),
                                                 real_test_recall=round(ref[m][c], 4)))

                # 2. nearest-neighbour copy check (once per source/nz/fold)
                for c in range(7):
                    si = np.where(ys == c)[0]; ri = np.where(yr == c)[0]
                    if len(si) == 0 or len(ri) < 2:
                        continue
                    si = rng.choice(si, min(a.nn_per_class, len(si)), replace=False)
                    rq = rng.choice(ri, min(a.nn_per_class, len(ri)), replace=False)
                    Fs = features(Xs[si], fx); Fc = Fr[ri]
                    s_sim, s_idx = (Fs @ Fc.T).max(1)
                    rr = Fr[rq] @ Fc.T
                    rr[torch.arange(len(rq)), torch.from_numpy(np.searchsorted(ri, rq))] = -2   # exclude itself
                    r_sim, r_idx = rr.max(1)
                    s_ssim = [ssim(Xs[i], Xr[ri[j]]) for i, j in zip(si, s_idx.numpy())]
                    r_ssim = [ssim(Xr[i], Xr[ri[j]]) for i, j in zip(rq, r_idx.numpy())]
                    nn_rows.append(dict(fold=fold, source=src, n_z=nz, cls=NAMES[c], n=len(si),
                                        syn_nn_cos=round(float(s_sim.mean()), 4), real_nn_cos=round(float(r_sim.mean()), 4),
                                        syn_nn_ssim=round(float(np.mean(s_ssim)), 4), real_nn_ssim=round(float(np.mean(r_ssim)), 4),
                                        syn_frac_ssim_gt_0_95=round(float(np.mean(np.array(s_ssim) > 0.95)), 4),
                                        real_frac_ssim_gt_0_95=round(float(np.mean(np.array(r_ssim) > 0.95)), 4)))
                del Xs; gc.collect(); torch.cuda.empty_cache()
        del nets, Fr, Xr, Xt; gc.collect(); torch.cuda.empty_cache()

        sfx = f"_{a.tag}" if a.tag else ""
        for name, rows in ((f"r22_label_consistency{sfx}.csv", lab_rows), (f"r22_nearest_neighbour{sfx}.csv", nn_rows)):
            if rows:
                with open(OUT / name, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"fold {fold} done, results saved")

    if lab_rows:
        import pandas as pd
        L = pd.DataFrame(lab_rows)
        print("\nLabel agreement on synthetic images (mean over folds) vs recall on real test images:")
        print(L.groupby(["source", "model", "cls"])[["agreement", "real_test_recall"]].mean().round(3).unstack("source").to_string())
    if nn_rows:
        import pandas as pd
        N = pd.DataFrame(nn_rows)
        print("\nNearest real neighbour (mean over folds): synthetic vs real-real reference")
        print(N.groupby(["source", "n_z", "cls"])[["syn_nn_cos", "real_nn_cos", "syn_nn_ssim", "real_nn_ssim",
                                                    "syn_frac_ssim_gt_0_95", "real_frac_ssim_gt_0_95"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
