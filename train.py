"""
PhishGuard - train and evaluate the models.

Run:  python train.py
Writes: model/phishguard.joblib, results/metrics.json and results/*.png
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import feature_names

SEED = 42
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load data (downloaded automatically on first run) ---------------------
DATA_PATH = "data/dataset_full.csv"
DATA_URL = ("https://raw.githubusercontent.com/GregaVrbancic/"
            "Phishing-Dataset/master/dataset_full.csv")
if not os.path.exists(DATA_PATH):
    import urllib.request
    os.makedirs("data", exist_ok=True)
    print("Downloading dataset (about 25 MB)...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
df = pd.read_csv(DATA_PATH)
FEATS = feature_names()
X, y = df[FEATS], df["phishing"]
print(f"Dataset: {len(df)} URLs | phishing={int(y.sum())} legitimate={int((y == 0).sum())}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 2. Candidate models -----------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25,
                                            n_jobs=-1, random_state=SEED),
}

results = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(X_tr, y_tr)
    train_s = time.time() - t0
    prob = m.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    # single-URL latency (average over 200 one-row predictions)
    rows = X_te.iloc[:200]
    t0 = time.time()
    for i in range(len(rows)):
        m.predict_proba(rows.iloc[[i]])
    latency_ms = (time.time() - t0) / len(rows) * 1000
    joblib.dump(m, "model/_tmp.joblib")
    size_mb = os.path.getsize("model/_tmp.joblib") / 1e6
    results[name] = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, prob),
        "train_seconds": train_s,
        "latency_ms": latency_ms,
        "model_mb": size_mb,
    }
    print(f"{name:20s} acc={results[name]['accuracy']:.4f} f1={results[name]['f1']:.4f} "
          f"auc={results[name]['roc_auc']:.4f} latency={latency_ms:.2f}ms size={size_mb:.1f}MB")
os.remove("model/_tmp.joblib")

# 3. Pick best by F1 and save ------------------------------------------------
best = max(results, key=lambda k: results[k]["f1"])
best_model = models[best]
joblib.dump({"model": best_model, "features": FEATS, "name": best}, "model/phishguard.joblib")
print(f"\nBest model: {best} -> model/phishguard.joblib")

# 4. Baseline: same RF using ALL 111 features (incl. DNS/WHOIS lookups) ---
all_feats = [c for c in df.columns if c != "phishing"]
Xa_tr, Xa_te = X_tr.index, X_te.index
rf_all = RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=SEED)
rf_all.fit(df.loc[Xa_tr, all_feats], y_tr)
pa = rf_all.predict(df.loc[Xa_te, all_feats])
full_baseline = {"accuracy": accuracy_score(y_te, pa), "f1": f1_score(y_te, pa),
                 "n_features": len(all_feats)}
print(f"Full-feature RF ({len(all_feats)} features, needs network lookups): "
      f"acc={full_baseline['accuracy']:.4f} f1={full_baseline['f1']:.4f}")

# 5. Save metrics ---------------------------------------------------------
cm = confusion_matrix(y_te, best_model.predict(X_te))
json.dump({"dataset_size": len(df), "phishing": int(y.sum()), "legitimate": int((y == 0).sum()),
           "train_size": len(X_tr), "test_size": len(X_te), "n_features": len(FEATS),
           "best_model": best, "models": results, "confusion_matrix": cm.tolist(),
           "full_feature_baseline": full_baseline},
          open("results/metrics.json", "w"), indent=2)

# 6. Plots ------------------------------------------------------------------
plt.rcParams.update({"font.size": 11})

# Model comparison
names = list(results)
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(names)); w = 0.2
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, met in enumerate(metrics):
    ax.bar(x + (i - 1.5) * w, [results[n][met] * 100 for n in names], w, label=met.capitalize())
ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
ax.set_ylabel("Score (%)"); ax.set_ylim(50, 100); ax.legend(ncol=4, loc="lower right")
ax.set_title("Model comparison on test set"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("results/model_comparison.png", dpi=200); plt.close(fig)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Legitimate", "Phishing"]); ax.set_yticklabels(["Legitimate", "Phishing"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix ({best})")
fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=200); plt.close(fig)

# ROC curves
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for n, m in models.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    ax.plot(fpr, tpr, label=f"{n} (AUC={results[n]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/roc_curves.png", dpi=200); plt.close(fig)

# Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(imp.index, imp.values, color="#2b6cb0")
ax.set_xlabel("Importance"); ax.set_title("Top 10 features (Random Forest)")
fig.tight_layout(); fig.savefig("results/feature_importance.png", dpi=200); plt.close(fig)
print("Plots saved in results/")
