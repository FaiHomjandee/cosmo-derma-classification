#!/usr/bin/env python
"""Revision analyses from the saved ablation predictions (all 5 folds, CPU only).
Outputs in revision_analysis/:
  seen_unseen.csv     macro F1 / G-mean / MEL & DF recall, all vs unseen-lesion test images
  perclass.csv        per-class precision/recall/specificity/F1/AUC/AP (pooled over folds)
  auc_summary.csv     macro AUC, micro AUC, macro AP, ECE
  bootstrap.csv       CoSMO - baseline, lesion-level cluster bootstrap 95% CI, Holm-adjusted p
"""
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

B = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"; OUT.mkdir(exist_ok=True)
L = B / "data" / "metadata"
NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
NZ = {"resnet18": 128, "efficientnet_b0": 512, "mobilenet_v2": 512, "vit": 128, "vim_real": 512}
TAG = {"imb_noaug": ("imbalanced_data_noaugment", 0), "aug_only": ("imbalanced_data", 0), "ros": ("ros", 0),
       "ae_smote": ("ae_smote_alpha1.0", 1), "cosmo": ("balanced_data", 1)}
# (results root, model name in the CSVs, analysis label, Holm family)
# main = five architectures all trained from scratch; ref = ImageNet-pretrained Vim, reported separately
RUNS = [("outputs/ablation", "resnet18", "resnet18", "main"), ("outputs/ablation", "mobilenet_v2", "mobilenet_v2", "main"),
        ("outputs/ablation", "efficientnet_b0", "efficientnet_b0", "main"), ("outputs/ablation", "vit", "vit", "main"),
        ("outputs/ablation_vim_scratch", "vim_real", "vim", "main"),
        ("outputs/ablation", "vim_real", "vim_pretrained", "ref")]

d = np.load(Path.home() / ".medmnist" / "dermamnist.npz")
y_all = np.concatenate([d["train_labels"], d["val_labels"], d["test_labels"]]).ravel()
info = list(csv.DictReader(open(L / "dermamnist_split_info.csv")))
meta = {r["image_id"]: r for r in csv.DictReader(open(L / "combined_metadata_original.csv"))}
img = []
for s in ("train", "val", "test"):
    img += [r["image_id"] for r in sorted([r for r in info if r["split"] == s], key=lambda r: int(r["index"]))]
les_all = np.array([meta[i]["lesion_id"] for i in img])
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))
test_les, unseen = [], []
for tr, te in FOLDS:
    trn = tr[: int(len(tr) * 0.9)]
    test_les.append(les_all[te]); unseen.append(~np.isin(les_all[te], list(set(les_all[trn]))))
test_les = np.concatenate(test_les); unseen = np.concatenate(unseen)

def load(root, model, cond):
    tag, syn = TAG[cond]
    z = NZ[model] if syn else 0
    Y, P, Pr = [], [], []
    for k, (tr, te) in enumerate(FOLDS, 1):
        f = np.load(B / root / "predictions" / f"pred_{tag}_{model}_z{z}_fold{k}.npz")
        assert (f["y_true"] == y_all[te]).all()
        Y.append(f["y_true"]); P.append(f["y_pred"]); Pr.append(f["prob"])
    return np.concatenate(Y), np.concatenate(P), np.concatenate(Pr)

def cm(y, p):
    return np.bincount(y * 7 + p, minlength=49).reshape(7, 7).astype(float)

def f1_gm(c):
    tp = np.diag(c); rec = tp / np.maximum(c.sum(1), 1); prec = tp / np.maximum(c.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    present = c.sum(1) > 0
    return f1[present].mean(), float(np.prod(rec[present]) ** (1 / present.sum()))

def ece(y, pr, bins=15):
    conf, pred = pr.max(1), pr.argmax(1); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

seen_rows, pc_rows, auc_rows, boot_rows = [], [], [], []
data = {}
for root, model, label, fam in RUNS:
    for cond in TAG:
        y, p, pr = load(root, model, cond); data[(label, cond)] = (y, p, pr)
        for sub, m in (("all", np.ones(len(y), bool)), ("unseen", unseen)):
            c = cm(y[m], p[m]); f1, g = f1_gm(c); rec = np.diag(c) / np.maximum(c.sum(1), 1)
            seen_rows.append(dict(model=label, condition=cond, subset=sub, n=int(m.sum()), F1=round(f1, 4), GM=round(g, 4),
                                  recall_MEL=round(rec[4], 3), recall_DF=round(rec[3], 3), n_MEL=int(c[4].sum()), n_DF=int(c[3].sum())))
        c = cm(y, p); aucs, aps = [], []
        for k, n in enumerate(NAMES):
            tp, fn, fp = c[k, k], c[k].sum() - c[k, k], c[:, k].sum() - c[k, k]; tn = c.sum() - tp - fn - fp
            pos = (y == k).astype(int); auc = roc_auc_score(pos, pr[:, k]); ap = average_precision_score(pos, pr[:, k])
            aucs.append(auc); aps.append(ap)
            prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn)
            pc_rows.append(dict(model=label, condition=cond, cls=n, n=int(tp + fn), precision=round(prec, 3), recall=round(rec, 3),
                                specificity=round(tn / (tn + fp), 4), F1=round(2 * prec * rec / (prec + rec) if prec + rec else 0, 3),
                                AUC=round(auc, 4), AP=round(ap, 4)))
        auc_rows.append(dict(model=label, condition=cond, macro_AUC=round(np.mean(aucs), 4),
                             micro_AUC=round(roc_auc_score(np.eye(7)[y].ravel(), pr.ravel()), 4),
                             macro_AP=round(np.mean(aps), 4), ECE=round(ece(y, pr), 4), mean_conf=round(pr.max(1).mean(), 4)))

# lesion-level cluster bootstrap: CoSMO vs each baseline
rng = np.random.default_rng(0)
uniq, inv = np.unique(test_les, return_inverse=True)
groups = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
NB = 2000
for root, model, label, fam in RUNS:
    y, pc, _ = data[(label, "cosmo")]
    for base in ("imb_noaug", "aug_only", "ae_smote"):
        _, pb, _ = data[(label, base)]
        fa, ga = f1_gm(cm(y, pc)); fb, gb = f1_gm(cm(y, pb))
        dF, dG = [], []
        for _ in range(NB):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            a = f1_gm(cm(y[idx], pc[idx])); b = f1_gm(cm(y[idx], pb[idx]))
            dF.append(a[0] - b[0]); dG.append(a[1] - b[1])
        dF, dG = np.array(dF), np.array(dG)
        pF = min(1.0, 2 * min((dF <= 0).mean(), (dF >= 0).mean())); pG = min(1.0, 2 * min((dG <= 0).mean(), (dG >= 0).mean()))
        boot_rows.append(dict(model=label, family=fam, comparison=f"cosmo-{base}", dF1=round(fa - fb, 4),
                              F1_lo=round(np.percentile(dF, 2.5), 4), F1_hi=round(np.percentile(dF, 97.5), 4), p_F1=pF,
                              dGM=round(ga - gb, 4), GM_lo=round(np.percentile(dG, 2.5), 4), GM_hi=round(np.percentile(dG, 97.5), 4), p_GM=pG))
# Holm across all comparisons, separately for F1 and GM
# Holm within each family (main: 5 models x 3 comparisons = 15; ref: 3), separately for F1 and GM
for fam in ("main", "ref"):
    idx_f = [i for i, r in enumerate(boot_rows) if r["family"] == fam]
    for key in ("p_F1", "p_GM"):
        order = sorted(idx_f, key=lambda i: boot_rows[i][key]); n = len(order); prev = 0
        for rank, i in enumerate(order):
            adj = min(1.0, max(prev, (n - rank) * boot_rows[i][key])); boot_rows[i][key + "_holm"] = round(adj, 4); prev = adj

for name, rows in (("seen_unseen.csv", seen_rows), ("perclass.csv", pc_rows), ("auc_summary.csv", auc_rows), ("bootstrap.csv", boot_rows)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("unseen test images:", int(unseen.sum()), "of", len(unseen), "| lesions in test pool:", len(uniq))
print("done")
