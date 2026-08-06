#!/usr/bin/env python
# coding: utf-8
"""
CoSMO: Contrastive Synthetic Minority Oversampling
Stage 2 -- Downstream classification

Loads the fold-aligned balanced_data checkpoints produced by
run_sampling.py and trains/evaluates five classifier architectures
(ResNet-18, EfficientNet-B0, MobileNet-V2, ViT, ViM) using 5-fold
cross-validation.  Results are saved as a CSV with per-fold and
average metrics (F1, G-mean, AUC, ACC, balanced accuracy).

Usage:
    # CoSMO condition
    python run_classification.py --tag balanced_data --n_z 512 \\
        --data_dir ./data --balanced_dir ./outputs/balanced

    # Aug-only baseline
    python run_classification.py --tag imbalanced_data \\
        --data_dir ./data

    # No-augmentation baseline
    python run_classification.py --tag imbalanced_data_noaugment \\
        --data_dir ./data
"""

import os
import itertools
import time
import argparse
from pathlib import Path

import pandas as pd
import torch

from Derma_classification_lowmem import derma_classification


# ── Argument parsing ───────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="CoSMO: Stage 2 -- Classification")
    parser.add_argument("--data_dir",     type=str, default="./data",
                        help="Directory containing per-fold data files")
    parser.add_argument("--balanced_dir", type=str, default="./outputs/balanced",
                        help="Directory with balanced_data checkpoints (Stage 1 output)")
    parser.add_argument("--out_dir",      type=str, default="./outputs/results",
                        help="Directory to save result CSVs")
    parser.add_argument("--tag",          type=str, default="balanced_data",
                        choices=["balanced_data", "imbalanced_data",
                                 "imbalanced_data_noaugment"],
                        help="Experiment condition")
    parser.add_argument("--n_z",          type=int, nargs="+",
                        default=[64, 128, 256, 512],
                        help="Latent dimension(s) to evaluate (balanced_data only)")
    parser.add_argument("--n_folds",      type=int, default=5)
    parser.add_argument("--epochs",       type=int, default=100,
                        help="Classifier training epochs (default: 100)")
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--temperature",  type=float, default=0.07)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset",      type=str, default="dermamnist")
    parser.add_argument("--models",       type=str, nargs="+",
                        default=["resnet18", "efficientnet_b0", "mobilenet_v2",
                                 "vit", "vim"],
                        help="Architectures to evaluate")
    parser.add_argument("--gpu",          type=str, default="0")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    args_cli = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu

    DATA_DIR     = Path(args_cli.data_dir)
    BALANCED_DIR = Path(args_cli.balanced_dir)
    OUT_DIR      = Path(args_cli.out_dir);  OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_args = {
        "dim_h":              128,
        "n_channel":          3,
        "num_class":          7,
        "batch_size":         args_cli.batch_size,
        "img_size":           224,
        "classifier_epochs":  args_cli.epochs,
    }

    # For baselines, n_z is irrelevant (no balanced checkpoint needed);
    # use a dummy list of one value so the outer loop runs once.
    nz_list = args_cli.n_z if args_cli.tag == "balanced_data" else [None]

    for n_z in nz_list:
        params = {
            "lr":           args_cli.lr,
            "alpha":        args_cli.alpha,
            "temperature":  args_cli.temperature,
            "n_z":          n_z,
            "weight_decay": args_cli.weight_decay,
            "dataset_name": args_cli.dataset,
            "experiment_tag": args_cli.tag,
        }

        tag_str = args_cli.tag if n_z is None else f"{args_cli.tag}_nz{n_z}"
        models_str = "_".join(args_cli.models)
        results_path = OUT_DIR / f"{args_cli.dataset}_{tag_str}_{models_str}.csv"

        print(f"\n{'='*60}")
        print(f"Condition: {args_cli.tag}  |  n_z: {n_z}")
        print(f"{'='*60}")

        # ── Resume: skip folds already in the CSV ──────────────────────
        completed = {}
        results_list = []
        if results_path.exists():
            df_ex = pd.read_csv(results_path)
            df_ex = df_ex[df_ex["Fold"] != "Average"]
            for fold_num, fold_df in df_ex.groupby("Fold"):
                if set(fold_df["Model"]) >= set(args_cli.models):
                    completed[int(fold_num)] = fold_df.to_dict("records")
                    results_list.extend(completed[int(fold_num)])
                    print(f"Resuming: fold {fold_num} already complete, skipping.")

        fold_tracker = []

        for fold_idx in range(args_cli.n_folds):
            fold_num = fold_idx + 1

            if fold_num in completed:
                fold_tracker.extend(completed[fold_num])
                continue

            # Load per-fold data
            fold_path = DATA_DIR / f"{args_cli.dataset}_fold{fold_num}_data.pth"
            if not fold_path.exists():
                # Fallback: try loading the full folds_data.pth
                full_path = DATA_DIR / f"{args_cli.dataset}_folds_data.pth"
                if not full_path.exists():
                    print(f"ERROR: no fold data found at {fold_path} or {full_path}")
                    continue
                all_folds = torch.load(full_path, weights_only=False)
                train_loader_ae, cv_val_loader, test_loader = all_folds[fold_idx]
            else:
                train_loader_ae, cv_val_loader, test_loader = torch.load(
                    fold_path, weights_only=False)

            # ── Prepare training data ──────────────────────────────────
            if args_cli.tag == "balanced_data":
                fname = (f"{args_cli.dataset}_fold{fold_num}"
                         f"_lr{params['lr']}_alpha{params['alpha']}_z{n_z}.pth")
                ckpt_path = BALANCED_DIR / fname
                if not ckpt_path.exists():
                    print(f"Missing checkpoint: {ckpt_path}, skipping fold {fold_num}.")
                    continue
                ckpt = torch.load(ckpt_path, weights_only=False)
                X_train = ckpt["balanced_data"]
                y_train = ckpt["balanced_labels"].view(-1)
                print(f"\n[BALANCED] Fold {fold_num} | Train: {X_train.shape[0]}")
                del ckpt

            else:
                imgs_list, labs_list = [], []
                for imgs, labs in train_loader_ae:
                    imgs_list.append(imgs)
                    labs_list.append(labs.view(-1))
                X_train = torch.cat(imgs_list)
                y_train = torch.cat(labs_list)
                print(f"\n[{args_cli.tag.upper()}] Fold {fold_num} | Train: {X_train.shape[0]}")
                print(f"  Distribution: {torch.bincount(y_train)}")

            # ── Train & evaluate ───────────────────────────────────────
            results = derma_classification(
                params=params,
                args=model_args,
                model_names=args_cli.models,
                balanced_data=X_train,
                balanced_labels=y_train,
                val_loader=cv_val_loader,
                test_loader=test_loader,
            )
            del X_train, y_train

            for model_name, metrics in results.items():
                row = params.copy()
                row.update({
                    "Fold":  fold_num,
                    "Model": model_name,
                    "F1":    metrics.get("f1",    0.0),
                    "ASCA":  metrics.get("asca",  0.0),
                    "GM":    metrics.get("gmean", 0.0),
                    "AUC":   metrics.get("auc",   0.0),
                    "ACC":   metrics.get("acc",   0.0),
                })
                results_list.append(row)
                fold_tracker.append(row)

            pd.DataFrame(results_list).to_csv(results_path, index=False)

        # ── Fold averages ──────────────────────────────────────────────
        df_t = pd.DataFrame(fold_tracker)
        if not df_t.empty:
            print(f"\nSummary [{tag_str}]:")
            for model_name in df_t["Model"].unique():
                mdf = df_t[df_t["Model"] == model_name]
                avg = params.copy()
                avg.update({
                    "Fold":    "Average",
                    "Model":   model_name,
                    "F1":      mdf["F1"].mean(),
                    "ASCA":    mdf["ASCA"].mean(),
                    "GM":      mdf["GM"].mean(),
                    "AUC":     mdf["AUC"].mean(),
                    "ACC":     mdf["ACC"].mean(),
                    "Std_F1":  mdf["F1"].std(),
                    "Std_ACC": mdf["ACC"].std(),
                })
                results_list.append(avg)
                print(f"  {model_name:18}: "
                      f"F1={avg['F1']:.4f}±{avg['Std_F1']:.4f} | "
                      f"ACC={avg['ACC']:.4f}±{avg['Std_ACC']:.4f}")
            pd.DataFrame(results_list).to_csv(results_path, index=False)

    print("\nDone. Results saved to", OUT_DIR)


if __name__ == "__main__":
    main()
