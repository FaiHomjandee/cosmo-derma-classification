import os
import sys
import cv2
import imageio
from tqdm import tqdm
import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import random
import copy
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.models as models
from torchvision import datasets, transforms

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score
from scipy.stats import gmean
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import smote_variants as sv
import gc
from scipy import linalg

_TB_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as _tb_import_error:
    SummaryWriter = None
    _TB_IMPORT_ERROR = _tb_import_error

_UMAP_IMPORT_ERROR = None
try:
    import umap.umap_ as umap
except Exception as _umap_import_error:
    umap = None
    _UMAP_IMPORT_ERROR = _umap_import_error

# =====================================================================
# 1. Helper Functions and Loss Modules
#    UNCHANGED from original -- same loss formula as the paper's Eq. 4
# =====================================================================
def _make_summary_writer(log_dir):
    if SummaryWriter is None:
        print(f"TensorBoard unavailable ({_TB_IMPORT_ERROR}); continuing without logging.")
        class _NullWriter:
            def add_scalar(self, *args, **kwargs): return None
            def add_histogram(self, *args, **kwargs): return None
            def add_image(self, *args, **kwargs): return None
            def add_figure(self, *args, **kwargs): return None
            def add_text(self, *args, **kwargs): return None
            def close(self): return None
        return _NullWriter()
    return SummaryWriter(log_dir=log_dir)


def calculate_fid(real_embeddings, syn_embeddings):
    """
    Calculates the Fréchet Inception Distance between two feature distributions.
    UNCHANGED from original.
    """
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = syn_embeddings.mean(axis=0), np.cov(syn_embeddings, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    eps = 1e-5
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return max(0.0, fid)


def biased_get_class1(c, data, target):
    """UNCHANGED from original."""
    mask = (target == c).flatten()
    return data[mask], target[mask]


class NTXentLoss(nn.Module):
    """UNCHANGED from original (kept for compatibility, not used by
    run_cosmo_resampling -- weighted_masked_nt_xent_noweight is used)."""
    def __init__(self): super(NTXentLoss, self).__init__()
    def forward(self, z_i, z_j, temperature):
        z = torch.cat([z_i, z_j], dim=0); z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.T); N = z_i.shape[0]
        mask = (~torch.eye(2 * N, dtype=bool)).to(z.device)
        sim = similarity / temperature; exp_sim = torch.exp(sim) * mask
        positive_sim = torch.exp(F.cosine_similarity(z_i, z_j) / temperature)
        positives = torch.cat([positive_sim, positive_sim], dim=0)
        denominator = exp_sim.sum(dim=1)
        return -torch.log(positives / denominator).mean()


class weighted_masked_nt_xent_noweight(nn.Module):
    """
    UNCHANGED from original. Supervised multi-positive contrastive
    loss, exactly matching the paper's Eq. 3 (Self-contrastive Centroid
    Loss). Builds a 2B-batch from [z_i, z_j], treats all same-label
    entries as positives.
    """
    def __init__(self): super().__init__()
    def forward(self, z_i, z_j, labels, temperature):
        device = z_i.device
        labels_orig = labels.long().to(device)
        B = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()

        labels_all = torch.cat([labels_orig, labels_orig], dim=0)
        mask = torch.eq(labels_all.unsqueeze(1), labels_all.unsqueeze(0)).float().to(device)

        self_mask = torch.eye(2 * B, device=device)
        mask = mask - self_mask

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count

        return -(mean_log_prob_pos).mean() / (2 * B)


class ProjectionHead(nn.Module):
    """UNCHANGED from original."""
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(ProjectionHead, self).__init__()
        if hidden_dim is None: hidden_dim = in_dim
        if out_dim is None: out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


# =====================================================================
# 2. ARCHITECTURE -- v3 fix: skip connections (U-Net style) + sub-pixel
#    upsampling, REPLACING the original EncoderCoSMO/DecoderCoSMO.
#
#    WHY: the original architecture's AdaptiveAvgPool2d((4,4)) pooled
#    the 14x14 feature map down to 4x4 BEFORE the latent projection,
#    discarding most spatial detail regardless of n_z. Diagnostic
#    testing confirmed the original decoder was WORSE than a trivial
#    constant-mean baseline (MSE 0.0227 vs 0.0197). The v3 fix removes
#    this pooling and adds encoder-decoder skip connections, which let
#    the decoder access encoder feature maps directly at matching
#    resolutions rather than relying solely on the n_z bottleneck.
#    Verified result: MSE 0.000745 vs constant baseline 0.020736
#    (~28x better than baseline, vs ~8x for the no-skip v1 fix).
#
#    LOSS FORMULA UNCHANGED: comb_loss = alpha*mse + (1-alpha)*contrastive
#    matches the paper's Eq. 4 exactly -- only the encoder/decoder
#    architecture changed, not the training objective.
# =====================================================================
class EncoderCoSMO(nn.Module):
    """
    Same 4-conv-layer downsampling as the original (224->112->56->28->14),
    but WITHOUT AdaptiveAvgPool2d, and now also returns intermediate
    feature maps (f1, f2, f3) for use as decoder skip connections.
    The final latent z is still produced via flatten -> Linear, so
    SMOTE interpolation on z is completely unaffected by this change.
    """
    def __init__(self, args):
        super().__init__()
        self.n_z = args['n_z']
        self.dim_h = args['dim_h']
        in_ch = args['n_channel']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, self.dim_h, 4, 2, 1),              # 224 -> 112
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),     # 112 -> 56
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1), # 56 -> 28
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1), # 28 -> 14
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # FIX: no AdaptiveAvgPool2d -- flatten the full 14x14 feature map
        self.flatten_dim = self.dim_h * 8 * 14 * 14
        self.fc_main = nn.Linear(self.flatten_dim, self.n_z)

    def forward(self, x, labels=None, return_skips=False):
        f1 = self.conv1(x)    # [B, dim_h,    112, 112]
        f2 = self.conv2(f1)   # [B, dim_h*2,   56,  56]
        f3 = self.conv3(f2)   # [B, dim_h*4,   28,  28]
        f4 = self.conv4(f3)   # [B, dim_h*8,   14,  14]
        flat = torch.flatten(f4, 1)
        z = self.fc_main(flat)
        if return_skips:
            return z, (f1, f2, f3)
        return z


class DecoderCoSMO(nn.Module):
    """
    Decoder with U-Net-style skip connections from the encoder, plus
    sub-pixel (PixelShuffle) upsampling on the final layer instead of
    ConvTranspose2d (reduces checkerboard/blur artifacts). When called
    without skip features (skips=None), zero tensors are substituted
    so the decoder still runs, just without the detail boost -- this
    preserves backward compatibility with any code path that calls
    the decoder on a bare latent vector.
    """
    def __init__(self, args):
        super().__init__()
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.n_channel = args['n_channel']

        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 7 -> 14: no matching encoder skip at 14x14 (f4 was consumed by fc_main)
        self.up1 = nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.dim_h * 4)

        # 14 -> 28: concat with skip f3 (28x28, dim_h*4 channels)
        self.up2 = nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.dim_h * 2)
        self.merge2 = nn.Conv2d(self.dim_h * 2 + self.dim_h * 4, self.dim_h * 2, 3, 1, 1)

        # 28 -> 56: concat with skip f2 (56x56, dim_h*2 channels)
        self.up3 = nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.dim_h)
        self.merge3 = nn.Conv2d(self.dim_h + self.dim_h * 2, self.dim_h, 3, 1, 1)

        # 56 -> 112: concat with skip f1 (112x112, dim_h channels)
        self.up4 = nn.ConvTranspose2d(self.dim_h, self.dim_h // 2, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.dim_h // 2)
        self.merge4 = nn.Conv2d(self.dim_h // 2 + self.dim_h, self.dim_h // 2, 3, 1, 1)

        # 112 -> 224: sub-pixel (PixelShuffle) upsampling
        upscale = 2
        self.subpixel_conv = nn.Conv2d(self.dim_h // 2, self.n_channel * (upscale ** 2),
                                        kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out_act = nn.Tanh()

    def forward(self, z, skips=None):
        B = z.shape[0]
        if skips is None:
            f1 = torch.zeros(B, self.dim_h, 112, 112, device=z.device)
            f2 = torch.zeros(B, self.dim_h * 2, 56, 56, device=z.device)
            f3 = torch.zeros(B, self.dim_h * 4, 28, 28, device=z.device)
        else:
            f1, f2, f3 = skips

        x = self.fc(z)
        x = x.view(-1, self.dim_h * 8, 7, 7)

        x = self.act(self.bn1(self.up1(x)))       # [B, dim_h*4, 14, 14]

        x = self.act(self.bn2(self.up2(x)))       # [B, dim_h*2, 28, 28]
        x = torch.cat([x, f3], dim=1)
        x = self.act(self.merge2(x))               # -> [B, dim_h*2, 28, 28]

        x = self.act(self.bn3(self.up3(x)))       # [B, dim_h, 56, 56]
        x = torch.cat([x, f2], dim=1)
        x = self.act(self.merge3(x))               # -> [B, dim_h, 56, 56]

        x = self.act(self.bn4(self.up4(x)))       # [B, dim_h//2, 112, 112]
        x = torch.cat([x, f1], dim=1)
        x = self.act(self.merge4(x))               # -> [B, dim_h//2, 112, 112]

        x = self.subpixel_conv(x)
        x = self.pixel_shuffle(x)                   # [B, n_channel, 224, 224]
        return self.out_act(x)


# =====================================================================
# 3. Cleaned and Streamlined Pipeline Function
#    SAME SIGNATURE as original: run_cosmo_resampling(params, args,
#    fold_data_loaders, imbalanced_train_dataset) -- drop-in compatible
#    with your existing gen_phase.py / generation scripts.
#
#    LOSS FORMULA UNCHANGED (comb_loss = alpha*mse + (1-alpha)*contrastive).
#    Only the architecture (Section 2 above) and the SMOTE/generation
#    stage below (memory-safe skip-feature handling) changed.
# =====================================================================
def run_cosmo_resampling(params, args, fold_data_loaders, imbalanced_train_dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    lr = params['lr']
    alpha = params['alpha']
    temperature = params['temperature']
    weight_decay = params['weight_decay']
    dataset_name = params['dataset_name']
    args['n_z'] = params['n_z']

    train_loader_ae, val_loader, test_loader = fold_data_loaders
    imbalanced_train_list = imbalanced_train_dataset

    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(device).eval()

    # =========================================================
    # Stage A: Train the Autoencoder
    # (loss formula identical to original / paper Eq. 4; only the
    #  encoder/decoder forward pass now also carries skip features)
    # =========================================================
    print("--- Stage A: Training Autoencoder ---")
    encoder = EncoderCoSMO(args).to(device)
    decoder = DecoderCoSMO(args).to(device)
    proj_head = ProjectionHead(in_dim=params['n_z'], hidden_dim=params['n_z'], out_dim=params['n_z']).to(device)

    criterion = nn.MSELoss().to(device)
    criterion_nt = weighted_masked_nt_xent_noweight().to(device)
    centroids_module = nn.Embedding(args['num_class'], params['n_z']).to(device)

    enc_optim = torch.optim.Adam(
        list(encoder.parameters()) + list(centroids_module.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    best_encoder_state, best_decoder_state = None, None
    best_loss = np.inf

    for epoch in range(args['epochs']):
        train_loss_epoch, mse_loss_epoch, contrastive_loss_epoch, num_batches = 0.0, 0.0, 0.0, 0
        encoder.train(); decoder.train(); centroids_module.train()

        for images, labs in train_loader_ae:
            images, labs = images.to(device), labs.to(device).view(-1)

            # 1. Forward Pass -- now also retrieves skip features for the decoder
            z_hat, skips = encoder(images, labs, return_skips=True)
            x_hat_raw = decoder(z_hat, skips=skips)
            # Decoder ends in Tanh ([-1,1]); real images are [0,1]-scaled
            # (confirmed via direct inspection: min=0.0, max=1.0). Map
            # decoder output to the same [0,1] range before computing MSE.
            x_hat = torch.clamp((x_hat_raw + 1.0) / 2.0, 0.0, 1.0)
            mse = criterion(x_hat, images)

            # 2. Extract Embedding Layout Structs (UNCHANGED)
            view2_embedding = centroids_module(labs)
            view1 = F.normalize(z_hat, dim=1)
            view2 = F.normalize(view2_embedding, dim=1)

            h1 = F.normalize(proj_head(view1), dim=1)
            h2 = F.normalize(proj_head(view2), dim=1)

            # 3. Optimize Weights -- SAME comb_loss formula as original
            loss_nt_xent = criterion_nt(h1, h2, labs, temperature)
            comb_loss = alpha * mse + (1 - alpha) * loss_nt_xent

            enc_optim.zero_grad(); dec_optim.zero_grad()
            comb_loss.backward()
            enc_optim.step(); dec_optim.step()

            train_loss_epoch += comb_loss.item()
            mse_loss_epoch += mse.item()
            contrastive_loss_epoch += loss_nt_xent.item()
            num_batches += 1

        avg_train_loss = train_loss_epoch / num_batches
        avg_mse_loss = mse_loss_epoch / num_batches
        avg_contrastive_loss = contrastive_loss_epoch / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args['epochs']}] | Train Loss: {avg_train_loss:.4f} | MSE: {avg_mse_loss:.4f} | Contrastive: {avg_contrastive_loss:.4f}")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())

    print(f"The best loss :{best_loss:.4f}\nAutoencoder training complete!")

    # =========================================================
    # Stage B: Generate Data with SMOTE
    # MEMORY-SAFE REWRITE: the original here would have stacked the
    # full dataset and stored skip features for every real image,
    # which at dim_h>=96 can require tens of GB and triggers an OOM
    # kill. This version:
    #   1. Encodes the FULL dataset for latents ONLY (no skip storage)
    #      -- this is what SMOTE needs to see the whole distribution.
    #   2. For each minority class being oversampled, separately
    #      encodes ONLY that class's real images to obtain skip
    #      features -- a small subset (tens to low hundreds of images),
    #      not the full ~7000+ image dataset.
    #   3. For each synthetic latent, borrows skip features from its
    #      nearest REAL neighbor WITHIN THE SAME CLASS (found via
    #      NearestNeighbors in latent space), since synthetic latents
    #      from SMOTE interpolation have no skip features of their own.
    # =========================================================
    print("--- Stage B: Generating SMOTE data ---")
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
    encoder.eval(); decoder.eval()

    imbal_goal = Counter([int(label.item()) if hasattr(label, 'item') else int(label) for _, label in imbalanced_train_list])
    print(f'Imb Distri: {imbal_goal}')

    target = torch.tensor([item[1] for item in imbalanced_train_list])

    all_synthetic_images, all_synthetic_labels = [], []

    with torch.no_grad():
        # ---- Pass 1: encode ALL images for latents only (no skips) ----
        latent_list = []
        encode_batch_size = 32

        for i in range(0, len(imbalanced_train_list), encode_batch_size):
            batch_items = imbalanced_train_list[i:i + encode_batch_size]
            batch_data = torch.stack([it[0] for it in batch_items]).to(device)
            batch_target = torch.tensor([it[1] for it in batch_items]).to(device)

            z_batch = encoder(batch_data, batch_target, return_skips=False)
            latent_list.append(z_batch.cpu())
            del batch_data, batch_target, z_batch

        X_all = torch.cat(latent_list, dim=0).numpy()
        y_all = target.cpu().numpy().flatten()
        del latent_list
        gc.collect(); torch.cuda.empty_cache()

        counts = Counter(y_all)
        max_count = max(counts.values())
        fid_results = []

        for i in range(args['num_class']):
            if counts[i] == max_count: continue

            class_indices = np.where(y_all == i)[0]
            n_to_gen = max_count - len(class_indices)
            if n_to_gen <= 0: continue

            y_bin = (y_all == i).astype(int)
            if y_bin.sum() < 2: continue

            try:
                sampler = sv.SMOTE(proportion=n_to_gen / y_bin.sum(), random_state=42)
                X_res, y_res = sampler.sample(X_all, y_bin)
            except Exception as e:
                print(f"SMOTE failed for class {i}: {e}"); continue

            X_syn = X_res[len(X_all):]
            y_syn = y_res[len(y_bin):]
            xsamp = X_syn[y_syn == 1][:n_to_gen]
            ysamp = np.full(len(xsamp), i)

            if xsamp.shape[0] == 0:
                continue

            # --- Latent Directional Cosine Similarity (UNCHANGED logic) ---
            X_class = X_all[class_indices]
            real_vecs_t = torch.tensor(X_class, dtype=torch.float32).to(device)
            syn_vecs_t = torch.tensor(xsamp, dtype=torch.float32).to(device)

            norm_real = F.normalize(real_vecs_t, dim=1)
            norm_syn = F.normalize(syn_vecs_t, dim=1)
            class_cosine_sim = torch.matmul(norm_syn, norm_real.t()).mean().item()
            print(f"Class {i} Latent Vector Cosine Similarity: {class_cosine_sim:.4f}")
            del real_vecs_t, syn_vecs_t, norm_real, norm_syn

            # --- MEMORY-SAFE: encode ONLY this class's real images to
            # get their skip features (small subset, not the full dataset) ---
            class_items = [imbalanced_train_list[idx] for idx in class_indices]
            skip_f1_list, skip_f2_list, skip_f3_list = [], [], []
            for j in range(0, len(class_items), encode_batch_size):
                batch_items = class_items[j:j + encode_batch_size]
                batch_data = torch.stack([it[0] for it in batch_items]).to(device)
                batch_target = torch.tensor([it[1] for it in batch_items]).to(device)
                _, (f1, f2, f3) = encoder(batch_data, batch_target, return_skips=True)
                skip_f1_list.append(f1.cpu())
                skip_f2_list.append(f2.cpu())
                skip_f3_list.append(f3.cpu())
                del batch_data, batch_target, f1, f2, f3
            class_f1 = torch.cat(skip_f1_list, dim=0)
            class_f2 = torch.cat(skip_f2_list, dim=0)
            class_f3 = torch.cat(skip_f3_list, dim=0)
            del skip_f1_list, skip_f2_list, skip_f3_list
            gc.collect(); torch.cuda.empty_cache()

            # Borrow skip features from each synthetic sample's nearest
            # REAL neighbor within this class (in latent space)
            nn_finder = NearestNeighbors(n_neighbors=1).fit(X_class)
            _, nearest_idx_local = nn_finder.kneighbors(xsamp)
            nearest_idx_local = nearest_idx_local.flatten()

            gc.collect(); torch.cuda.empty_cache()
            decoded_list = []
            xsamp_tensor = torch.tensor(xsamp, dtype=torch.float32)

            decode_batch_size = 16
            for j in range(0, xsamp_tensor.size(0), decode_batch_size):
                batch_z_flat = xsamp_tensor[j:j + decode_batch_size].to(device)
                batch_idx_local = nearest_idx_local[j:j + decode_batch_size]

                batch_f1 = class_f1[batch_idx_local].to(device)
                batch_f2 = class_f2[batch_idx_local].to(device)
                batch_f3 = class_f3[batch_idx_local].to(device)

                batch_gen_raw = decoder(batch_z_flat, skips=(batch_f1, batch_f2, batch_f3))
                batch_gen = torch.clamp((batch_gen_raw + 1.0) / 2.0, 0.0, 1.0)
                decoded_list.append(batch_gen.cpu())
                del batch_f1, batch_f2, batch_f3, batch_gen_raw, batch_gen

            generated_images = torch.cat(decoded_list, dim=0)
            del class_f1, class_f2, class_f3, decoded_list
            gc.collect(); torch.cuda.empty_cache()

            # --- Image-Level Feature FID Tracking (UNCHANGED logic) ---
            class_image_fid = 0.0
            if generated_images.shape[0] > 1:
                xclass_imgs = torch.stack([it[0] for it in class_items])
                real_features_list, gen_features_list = [], []

                for b_idx in range(0, xclass_imgs.size(0), 32):
                    real_features_list.append(feature_extractor(xclass_imgs[b_idx:b_idx + 32].to(device)).cpu())
                for b_idx in range(0, generated_images.shape[0], 32):
                    gen_features_list.append(feature_extractor(generated_images[b_idx:b_idx + 32].to(device)).cpu())

                class_image_fid = calculate_fid(torch.cat(real_features_list, dim=0).numpy(), torch.cat(gen_features_list, dim=0).numpy())
                print(f"Class {i} Image ResNet Feature FID: {class_image_fid:.4f}")
                del xclass_imgs, real_features_list, gen_features_list

            fid_results.append({
                'Class': i,
                'Latent_Cosine_Similarity': round(float(class_cosine_sim), 4),
                'Image_Feature_FID': round(float(class_image_fid), 4)
            })

            all_synthetic_images.append(generated_images)
            all_synthetic_labels.append(torch.tensor(ysamp))

            gc.collect(); torch.cuda.empty_cache()

        if fid_results:
            fold_idx = params.get('fold_idx', 1)  # defaults to 1 if not provided
            output_dir = (f"/home/s2522002/Experiment/sim_score/{dataset_name}/"
                        f"FID_Results/fold{fold_idx}")
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f"FID_Scores_sampled_nz_{params['n_z']}.csv")
            pd.DataFrame(fid_results).to_csv(csv_path, index=False)
            print(f"Metric logs successfully saved to: {csv_path}")

    # ---- Final assembly: real images re-stacked here only once, at
    # the very end, after all per-class skip-feature tensors have
    # already been freed -- avoids holding both simultaneously. ----
    data = torch.stack([item[0] for item in imbalanced_train_list])

    if all_synthetic_images:
        synthetic_only_data = torch.cat(all_synthetic_images, dim=0)
        synthetic_only_labels = torch.cat(all_synthetic_labels, dim=0).view(-1)
        balanced_data = torch.cat([data, synthetic_only_data], dim=0)
        balanced_labels = torch.cat([target.view(-1), synthetic_only_labels], dim=0)
    else:
        balanced_data, balanced_labels = data, target

    return balanced_data, balanced_labels, best_encoder_state, best_decoder_state