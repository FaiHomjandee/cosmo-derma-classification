# CoSMO: From Rare to Recognized

**Contrastive Latent Oversampling for Minority Skin Lesion Classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **CoSMO**, a two-stage framework for imbalanced dermatological image classification.  
*Paper under review at a peer-reviewed international journal.*

---

## Overview

CoSMO addresses severe class imbalance (up to 50:1 ratio in DermaMNIST) by:

1. **Stage 1 — Structured latent space**: Training a class-centroid convolutional autoencoder with U-Net-style skip connections using a composite loss that jointly optimizes reconstruction fidelity and class discriminability via a self-contrastive centroid objective.

2. **Stage 2 — Latent SMOTE**: Generating synthetic minority-class samples via SMOTE interpolation directly within the structured latent space, decoding each synthetic latent using skip features borrowed from its nearest real neighbour, and training downstream classifiers on the resulting balanced dataset.

<p align="center">
  <img src="assets/workflow.png" width="800" alt="CoSMO pipeline overview"/>
</p>

---

## Results (DermaMNIST, 5-fold CV)

| Model | Condition | ACC | AUC | F1 | G-mean |
|---|---|---|---|---|---|
| ViM (nz=512) | **CoSMO** | **0.8837** | 0.9568 | **0.8145** | **0.7924** |
| ViM (nz=512) | Aug-only | 0.8788 | **0.9666** | 0.7782 | 0.7426 |
| MobileNet-V2 (nz=64) | **CoSMO** | **0.7704** | 0.9031 | **0.5910** | **0.5457** |
| MobileNet-V2 (nz=64) | Aug-only | 0.7521 | **0.9091** | 0.4542 | 0.0000 |

CoSMO achieves the best F1 and G-mean for **every** architecture tested.  
Statistical significance: Friedman *p* = 0.0067; one-sided Wilcoxon *p* = 0.0312, *r* = 0.905.

---

## Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- smote-variants
- scikit-learn
- medmnist
- timm (for ViT / ViM backbones)
- scipy
- pandas
- tqdm

---

## Data Preparation

Download DermaMNIST via the [MedMNIST](https://medmnist.com/) package:

```python
import medmnist
from medmnist import DermaMNIST
DermaMNIST(split='train', download=True, root='./data/raw')
```

Then prepare the 5-fold cross-validation splits:

```bash
python prepare_folds.py --data_dir ./data/raw --out_dir ./data
```

This produces `dermamnist_folds_data.pth` and per-fold files
(`dermamnist_fold{1..5}_data.pth`) in `./data/`.

---

## Usage

### Stage 1 — Oversampling (CoSMO generation)

Train the autoencoder and generate balanced datasets for all four
latent dimensions across all five folds:

```bash
python run_sampling.py \
    --data_dir  ./data \
    --out_dir   ./outputs/balanced \
    --fid_dir   ./outputs/fid \
    --n_z 64 128 256 512 \
    --n_folds 5 \
    --epochs 200 \
    --gpu 0
```

Each completed fold/nz combination is saved immediately to
`./outputs/balanced/dermamnist_fold{N}_lr0.0002_alpha0.5_z{NZ}.pth`.
The script is **crash-resumable**: re-running skips any fold/nz whose
output file already exists.

FID and cosine similarity scores are saved per fold to
`./outputs/fid/fold{N}/FID_Scores_sampled_nz_{NZ}.csv`.

---

### Stage 2 — Classification

**CoSMO condition** (all four nz values):

```bash
python run_classification.py \
    --tag balanced_data \
    --n_z 64 128 256 512 \
    --data_dir ./data \
    --balanced_dir ./outputs/balanced \
    --out_dir ./outputs/results \
    --gpu 0
```

**Aug-only baseline:**

```bash
python run_classification.py \
    --tag imbalanced_data \
    --data_dir ./data \
    --out_dir ./outputs/results \
    --gpu 0
```

**Imb-noaug baseline:**

```bash
python run_classification.py \
    --tag imbalanced_data_noaugment \
    --data_dir ./data \
    --out_dir ./outputs/results \
    --gpu 0
```

Results are saved to `./outputs/results/` as CSV files with per-fold
and average metrics.

---

## Repository Structure

```
CoSMO/
├── run_sampling.py           # Stage 1: autoencoder + SMOTE generation
├── run_classification.py     # Stage 2: downstream classification
├── Derma_CoSMO_modified.py   # CoSMO model (encoder, decoder, loss, run_cosmo_resampling)
├── Derma_classification_lowmem.py  # Classifier training loop (all 5 architectures)
├── prepare_folds.py          # Data preparation utility
├── requirements.txt
├── assets/
│   └── workflow.png
└── README.md
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_z` | {64, 128, 256, 512} | Latent bottleneck dimension |
| `alpha` | 0.5 | Weight for reconstruction loss in L_total |
| `temperature` | 0.07 | Contrastive loss temperature τ |
| `dim_h` | 128 | Encoder base channel width |
| `epochs` | 200 | Autoencoder training epochs |
| `classifier_epochs` | 100 | Downstream classifier epochs |
| `batch_size` | 32 | Autoencoder batch size |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{cosmo2025,
  title   = {From Rare to Recognized: Contrastive Latent Oversampling
             for Minority Skin Lesion Classification},
  author  = {Homjandee, Suvaporn and Sinapiromsaran, Krung
             and Kertkeidkachorn, Natthawut},
  journal = {Under review},
  year    = {2025}
}
```

---

## Acknowledgements

The first author is supported by the Royal Thai Government's DPST
Scholarship. Computational resources were provided by the Department
of Mathematics and Computer Science, Chulalongkorn University, and
the Knowledge Laboratory (KnOWLab) at JAIST.
