#!/usr/bin/env python
"""
Build the 5-fold cross-validation splits used in the paper.

All official DermaMNIST splits (train + val + test, 10,015 images at 224x224)
are pooled and re-split with StratifiedKFold(5, shuffle=True, random_state=42).
Within each fold the training part is split 90/10 (by order) into
train / validation; the test part is the held-out fold.

Output (one file per fold, so later scripts load only the fold they need):
    data/prepared/dermamnist_fold{k}_data.pth  = [train_loader, val_loader, test_loader]

Note: folds are stratified by image, not by lesion. HAM10000 contains several
images of the same lesion, so a test image can share its lesion with a training
image. analysis/analyze_ablation.py reports results on unseen-lesion test images
separately (see the paper, Section 4.1).

    python prepare_folds.py --out_dir ./data/prepared
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import medmnist
from medmnist import INFO
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset


def make_loader(imgs, labels, shuffle):
    x = torch.from_numpy(imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data/prepared")
    ap.add_argument("--dataset", default="dermamnist")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    DataClass = getattr(medmnist, INFO[a.dataset]["python_class"])
    splits = [DataClass(split=s, size=224, download=True) for s in ("train", "val", "test")]
    imgs = np.concatenate([d.imgs for d in splits])
    labels = np.concatenate([d.labels for d in splits])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(skf.split(imgs, labels.ravel()), 1):
        path = out / f"{a.dataset}_fold{k}_data.pth"
        if path.exists():
            print(f"fold {k}: exists, skipping"); continue
        cut = int(len(tr) * 0.9)
        trn, val = tr[:cut], tr[cut:]
        torch.save([make_loader(imgs[trn], labels[trn], True),
                    make_loader(imgs[val], labels[val], False),
                    make_loader(imgs[te], labels[te], False)], path)
        print(f"fold {k}: train {len(trn)} / val {len(val)} / test {len(te)} -> {path}")


if __name__ == "__main__":
    main()
