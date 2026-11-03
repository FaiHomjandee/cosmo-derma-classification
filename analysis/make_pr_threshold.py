#!/usr/bin/env python
"""PR curves and threshold analysis for EfficientNet-B0 (Aug-only vs CoSMO), pooled out-of-fold predictions.
Writes fig9_pr_effnet.pdf/.png and pr_threshold.csv."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path(__file__).resolve().parent.parent; OUT = Path(__file__).resolve().parent / "results"; OUT.mkdir(exist_ok=True)
NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
d = np.load(Path.home() / ".medmnist" / "dermamnist.npz")
y_all = np.concatenate([d["train_labels"], d["val_labels"], d["test_labels"]]).ravel()
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))
def load(tag, z):
    Y, Pr = [], []
    for k, (tr, te) in enumerate(FOLDS, 1):
        f = np.load(B / "outputs" / "ablation" / "predictions" / f"pred_{tag}_efficientnet_b0_z{z}_fold{k}.npz")
        assert (f["y_true"] == y_all[te]).all(); Y.append(f["y_true"]); Pr.append(f["prob"])
    return np.concatenate(Y), np.concatenate(Pr)
R = {"Aug-only": load("imbalanced_data", 0), "CoSMO": load("balanced_data", 512)}
rows = []
for cond, (y, pr) in R.items():
    for c in ("MEL", "DF"):
        k = NAMES.index(c); t = (y == k); s = pr[:, k]; arg = pr.argmax(1) == k
        rows.append(dict(condition=cond, cls=c, rule="argmax", threshold=np.nan,
                         precision=(t & arg).sum() / max(arg.sum(), 1), recall=(t & arg).sum() / t.sum()))
        for th in (0.1, 0.2, 0.3, 0.5):
            p = s >= th
            rows.append(dict(condition=cond, cls=c, rule="score>=t", threshold=th,
                             precision=(t & p).sum() / max(p.sum(), 1), recall=(t & p).sum() / t.sum()))
        P, Rc, _ = precision_recall_curve(t, s)
        for rt in (0.5, 0.7):
            rows.append(dict(condition=cond, cls=c, rule=f"best precision at recall>={rt}", threshold=np.nan,
                             precision=P[Rc >= rt].max(), recall=rt))
        rows.append(dict(condition=cond, cls=c, rule="AP", threshold=np.nan, precision=average_precision_score(t, s), recall=np.nan))
        rows.append(dict(condition=cond, cls=c, rule="AUC", threshold=np.nan, precision=roc_auc_score(t, s), recall=np.nan))
df = pd.DataFrame(rows).round(3); df.to_csv(OUT / "pr_threshold.csv", index=False); print(df.to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, c in zip(ax, ("MEL", "DF")):
    k = NAMES.index(c)
    for cond, col in (("Aug-only", "#4D4D4D"), ("CoSMO", "#B2182B")):
        y, pr = R[cond]; t = y == k
        P, Rc, _ = precision_recall_curve(t, pr[:, k])
        a.plot(Rc, P, color=col, lw=2, label=f"{cond} (AP {average_precision_score(t, pr[:, k]):.3f})")
        am = pr.argmax(1) == k
        a.plot((t & am).sum() / t.sum(), (t & am).sum() / max(am.sum(), 1), "o", color=col, ms=8, mec="white")
    a.axhline(t.mean(), ls=":", color="gray", lw=1)
    a.set(title=f"{c} (EfficientNet-B0)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    a.grid(alpha=.3, ls="--"); a.legend(loc="upper right", fontsize=9)
plt.tight_layout()
for e in ("pdf", "png"): plt.savefig(OUT / f"fig9_pr_effnet.{e}", dpi=300, bbox_inches="tight")
