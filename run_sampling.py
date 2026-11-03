#!/usr/bin/env python
# coding: utf-8
"""
CoSMO: Contrastive Synthetic Minority Oversampling
Stage 1 -- Autoencoder training + SMOTE generation

Trains a class-centroid autoencoder with skip connections (U-Net style)
using a composite loss:
    L_total = alpha * L_rec + (1 - alpha) * L_con

Then generates balanced synthetic minority-class images via SMOTE
interpolation in the structured latent space, decoding each synthetic
latent using skip features borrowed from its nearest real neighbor.

Usage:
    python run_sampling.py --n_z 128 512                                   # CoSMO
    python run_sampling.py --n_z 128 512 --alpha 1.0 --out_dir ./outputs/balanced_ae_smote  # AE+SMOTE
"""

import os
import gc
import copy
import time
import argparse
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from scipy import linalg
import smote_variants as sv

from Derma_CoSMO_modified import run_cosmo_resampling


# ── Argument parsing ───────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="CoSMO: Stage 1 -- Oversampling")
    parser.add_argument("--data_dir",  type=str, default="./data/prepared",
                        help="Directory with dermamnist_fold{k}_data.pth (prepare_folds.py)")
    parser.add_argument("--out_dir",   type=str, default="./outputs/balanced",
                        help="Directory to save balanced_data checkpoints")
    parser.add_argument("--fid_dir",   type=str, default="./outputs/fid",
                        help="Directory to save FID/cosine CSV results")
    parser.add_argument("--n_z",       type=int, nargs="+",
                        default=[64, 128, 256, 512],
                        help="Latent dimension(s) to sweep (default: 64 128 256 512)")
    parser.add_argument("--n_folds",   type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--epochs",    type=int, default=200,
                        help="Autoencoder training epochs per fold (default: 200)")
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=2e-4)
    parser.add_argument("--alpha",     type=float, default=0.5,
                        help="Weight of L_rec (0.5 = CoSMO; 1.0 = AE+SMOTE ablation, no contrastive loss)")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dataset",   type=str, default="dermamnist")
    parser.add_argument("--gpu",       type=str, default="0",
                        help="CUDA device index (default: 0)")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    args_cli = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu

    DATA_DIR    = Path(args_cli.data_dir)
    OUT_DIR     = Path(args_cli.out_dir);   OUT_DIR.mkdir(parents=True, exist_ok=True)
    FID_DIR     = Path(args_cli.fid_dir);   FID_DIR.mkdir(parents=True, exist_ok=True)

    model_args = {
        "dim_h":      128,
        "n_channel":  3,
        "num_class":  7,
        "epochs":     args_cli.epochs,
        "batch_size": args_cli.batch_size,
        "img_size":   224,
    }

    # Fold files are loaded one at a time inside the loop (see prepare_folds.py)

    # ── Sweep over n_z and folds ───────────────────────────────────────
    for n_z in args_cli.n_z:
        params = {
            "lr":           args_cli.lr,
            "alpha":        args_cli.alpha,
            "temperature":  args_cli.temperature,
            "n_z":          n_z,
            "weight_decay": args_cli.weight_decay,
            "dataset_name": args_cli.dataset,
        }
        print(f"\n{'='*60}")
        print(f"n_z = {n_z}")
        print(f"{'='*60}")

        for fold_idx in range(args_cli.n_folds):
            fold_num = fold_idx + 1

            save_path = OUT_DIR / (
                f"{args_cli.dataset}_fold{fold_num}"
                f"_lr{params['lr']}_alpha{params['alpha']}_z{n_z}.pth"
            )

            # Skip if already done (crash-resume)
            if save_path.exists():
                print(f"  Fold {fold_num}/{args_cli.n_folds}: already done, skipping.")
                continue

            print(f"  Fold {fold_num}/{args_cli.n_folds}: training autoencoder + generating ...")
            t0 = time.time()

            params_fold = dict(params)
            params_fold["fold_idx"] = fold_num
            params_fold["fid_dir"]  = str(FID_DIR)

            fold_loaders = torch.load(DATA_DIR / f"{args_cli.dataset}_fold{fold_num}_data.pth",
                                      weights_only=False)
            ds = fold_loaders[0].dataset
            imbalanced = list(zip(ds.tensors[0], ds.tensors[1].view(-1)))

            balanced_data, balanced_labels, enc_state, dec_state = run_cosmo_resampling(
                params=params_fold,
                args=model_args,
                fold_data_loaders=fold_loaders,
                imbalanced_train_dataset=imbalanced,
            )
            del fold_loaders, imbalanced

            # Memory cleanup before saving
            gc.collect();  torch.cuda.empty_cache()

            for attempt in range(3):
                try:
                    torch.save({
                        "balanced_data":   balanced_data,
                        "balanced_labels": balanced_labels,
                        "encoder_state":   enc_state,
                        "decoder_state":   dec_state,
                        "source_real_fold": fold_num,
                    }, save_path)
                    print(f"  Saved → {save_path}  ({(time.time()-t0)/60:.1f} min)")
                    break
                except Exception as e:
                    print(f"  Save attempt {attempt+1}/3 failed: {e}")
                    if save_path.exists():
                        save_path.unlink()
                    gc.collect();  torch.cuda.empty_cache()
                    time.sleep(10)

            del balanced_data, balanced_labels, enc_state, dec_state
            gc.collect();  torch.cuda.empty_cache()

    print("\nAll sampling complete.")


if __name__ == "__main__":
    main()
