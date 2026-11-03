#!/usr/bin/env python
"""
CoSMO, Stage 2: train the downstream classifiers and save per-image test
predictions for every fold.

Every condition uses IDENTICAL classifier settings (AdamW, lr 1e-4,
weight decay 1e-2, label smoothing 0.1, 100 epochs, batch size 8). Validation
and test sets are always the real images of the fold.

Conditions (--conditions):
    imb_noaug   real imbalanced data, no augmentation
    aug_only    real imbalanced data + augmentation
    ae_smote    autoencoder + latent SMOTE without the contrastive loss
                (run_sampling.py --alpha 1.0 --out_dir outputs/balanced_ae_smote)
    cosmo       CoSMO (run_sampling.py --alpha 0.5 --out_dir outputs/balanced)
    ros         random oversampling of real minority images (supplementary only)

Latent dimension per architecture (Table 4): ResNet-18 / ViT 128, others 512.

Outputs:
    <out_dir>/results/results_<condition>.csv
    <out_dir>/predictions/pred_<tag>_<model>_z<nz>_fold<k>.npz  (y_true, y_pred, prob)

Examples:
    python run_classification.py --conditions imb_noaug aug_only ae_smote cosmo \
        --models resnet18 efficientnet_b0 mobilenet_v2 vit
    # Vision Mamba from scratch (needs the 'vimreal' env, see vim_real/)
    python run_classification.py --conditions imb_noaug aug_only ae_smote cosmo \
        --models vim_real --vim-scratch
    # ImageNet-pretrained Vision Mamba (reference row)
    python run_classification.py --conditions imb_noaug aug_only ae_smote cosmo --models vim_real
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

# The official Vim must be imported before smote_variants / OpenCV (CUDA library clash).
if "vim_real" in sys.argv:
    sys.path.insert(0, str(ROOT / "vim_real"))
    import vim_real_model  # noqa: F401

from Derma_classification_lowmem import derma_classification  # noqa: E402

DATA = "dermamnist"
MODELS = ["resnet18", "efficientnet_b0", "mobilenet_v2", "vit", "vim_real"]
BEST_NZ = {"resnet18": 128, "vit": 128, "efficientnet_b0": 512, "mobilenet_v2": 512, "vim_real": 512}
AE_LR = 0.0002  # autoencoder lr, part of the balanced-data file names
TAGS = {"imb_noaug": "imbalanced_data_noaugment", "aug_only": "imbalanced_data",
        "ros": "ros", "ae_smote": "ae_smote_alpha1.0", "cosmo": "balanced_data"}
ARGS = {"dim_h": 128, "n_channel": 3, "num_class": 7, "batch_size": 8, "img_size": 224,
        "classifier_epochs": 100}


def real_train(train_loader):
    ds = train_loader.dataset
    return ds.tensors[0], ds.tensors[1].view(-1)


def random_oversample(X, y, seed):
    rng = np.random.default_rng(seed)
    y_np = y.numpy(); n_max = np.bincount(y_np).max(); idx = []
    for c in np.unique(y_np):
        c_idx = np.where(y_np == c)[0]; idx.append(c_idx)
        if len(c_idx) < n_max:
            idx.append(rng.choice(c_idx, n_max - len(c_idx), replace=True))
    idx = torch.from_numpy(np.concatenate(idx))
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True, choices=list(TAGS))
    ap.add_argument("--models", nargs="+", default=MODELS[:4], choices=MODELS)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "prepared"))
    ap.add_argument("--cosmo_dir", default=str(ROOT / "outputs" / "balanced"))
    ap.add_argument("--ae_smote_dir", default=str(ROOT / "outputs" / "balanced_ae_smote"))
    ap.add_argument("--out_dir", default=None,
                    help="default: outputs/ablation (outputs/ablation_vim_scratch with --vim-scratch)")
    ap.add_argument("--clf-lr", type=float, default=1e-4)
    ap.add_argument("--clf-wd", type=float, default=1e-2)
    ap.add_argument("--vim-scratch", action="store_true",
                    help="train vim_real from random initialisation (the from-scratch Vim in Table 4)")
    ap.add_argument("--sweep-nz", type=int, default=None, choices=[64, 128, 256, 512],
                    help="use this n_z for every model (latent-dimension sweep, Section 5.9)")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU visible (torch.cuda.is_available() is False).")

    out = Path(a.out_dir) if a.out_dir else ROOT / "outputs" / ("ablation_vim_scratch" if a.vim_scratch else "ablation")
    if a.vim_scratch:
        os.environ["VIM_REAL_PRETRAINED"] = "0"
    if a.sweep_nz is not None:
        for m in a.models:
            BEST_NZ[m] = a.sweep_nz
        out = out / f"nz_sweep/nz{a.sweep_nz}"
    res_dir, pred_dir = out / "results", out / "predictions"
    res_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for cond in a.conditions:
        tag = TAGS[cond]
        synthetic = cond in ("ae_smote", "cosmo")
        res_path = res_dir / f"results_{cond}.csv"
        rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
        rows = [r for r in rows if float(r["clf_lr"]) == a.clf_lr and float(r["clf_wd"]) == a.clf_wd]
        done = {(int(r["Fold"]), r["Model"]) for r in rows}

        groups = {}
        for m in a.models:
            groups.setdefault(BEST_NZ[m] if synthetic else 0, []).append(m)

        for fold in a.folds:
            for nz, models in groups.items():
                todo = [m for m in models if (fold, m) not in done]
                if not todo:
                    continue
                print(f"\n== {cond} | fold {fold} | n_z {nz} | {todo}")
                train_loader, val_loader, test_loader = torch.load(
                    Path(a.data_dir) / f"{DATA}_fold{fold}_data.pth", weights_only=False)

                if cond in ("imb_noaug", "aug_only"):
                    X, y = real_train(train_loader)
                elif cond == "ros":
                    X, y = random_oversample(*real_train(train_loader), seed=42 + fold)
                else:
                    alpha = 0.5 if cond == "cosmo" else 1.0
                    src = Path(a.cosmo_dir if cond == "cosmo" else a.ae_smote_dir)
                    path = src / f"{DATA}_fold{fold}_lr{AE_LR}_alpha{alpha}_z{nz}.pth"
                    if not path.exists():
                        print(f"[MISSING] {path} - run run_sampling.py first"); continue
                    ck = torch.load(path, weights_only=False)
                    X, y = ck["balanced_data"], ck["balanced_labels"].view(-1); del ck

                torch.manual_seed(42 + fold); np.random.seed(42 + fold)
                params = {"lr": a.clf_lr, "weight_decay": a.clf_wd, "n_z": nz, "dataset_name": DATA,
                          "experiment_tag": tag, "fold": fold, "out_root": str(out), "pred_dir": str(pred_dir)}
                results = derma_classification(params=params, args=ARGS, model_names=todo,
                                               balanced_data=X, balanced_labels=y,
                                               val_loader=val_loader, test_loader=test_loader)
                del X, y, train_loader, val_loader, test_loader
                gc.collect(); torch.cuda.empty_cache()

                for m, met in results.items():
                    rows.append({"condition": cond, "experiment_tag": tag, "Fold": fold, "Model": m, "n_z": nz,
                                 "clf_lr": a.clf_lr, "clf_wd": a.clf_wd,
                                 "F1": met.get("f1", 0.0), "GM": met.get("gmean", 0.0),
                                 "AUC": met.get("auc", 0.0), "ACC": met.get("acc", 0.0),
                                 "val_F1": met.get("best_val_f1", float("nan"))})
                    done.add((fold, m))
                pd.DataFrame(rows).to_csv(res_path, index=False)

        if rows:
            df = pd.DataFrame(rows)
            print(df.groupby("Model")[["F1", "GM", "AUC", "ACC"]].agg(["mean", "std"]).round(4).to_string())
    print(f"\nDone in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
