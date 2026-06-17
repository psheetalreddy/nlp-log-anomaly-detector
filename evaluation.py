"""
evaluate.py — Offline evaluation of the trained AnomalyDetector
Uses cached models/anomaly_model.pkl (must exist — run main.py first to train).

Reads:
    logs/normal_logs.txt  → label 0 (normal)
    logs/anomaly_logs.txt  → label 1 (anomaly)

Two thresholds are reported:
    - Cached threshold : set at training time (5th percentile of normal scores)
    - Optimal threshold: derived from ROC curve (maximises TPR - FPR)

Outputs saved to: evaluation/
    confusion_matrix.png, roc_curve.png, pr_curve.png, score_distribution.png
"""

import os
import re
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)
from nlp_pipeline import get_embedding

MODEL_PATH   = "models/anomaly_model.pkl"
NORMAL_PATH  = "logs/normal_logs.txt"
ANOMALY_PATH = "logs/anomaly_logs.txt"
OUTPUT_DIR   = "evaluation"

_LINE_PATTERN = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\S]*\s+\S+\s+[\w/\-\.]+(?:\[\d+\])?:\s+(?P<message>.+)$'
)

def _extract_message(line: str) -> str:
    m = _LINE_PATTERN.match(line.strip())
    return m.group("message") if m else line.strip()

def load_lines(path: str) -> list[str]:
    with open(path, "r", errors="replace") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]

def embed_lines(lines: list[str]) -> np.ndarray:
    embeddings = []
    for i, line in enumerate(lines):
        emb = get_embedding(_extract_message(line))
        embeddings.append(emb)
        if (i + 1) % 20 == 0:
            print(f"  Embedded {i+1}/{len(lines)}")
    return normalize(np.array(embeddings, dtype=np.float32))

def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"[Evaluate] No cached model at {MODEL_PATH}. Run main.py first.")
        return

    model, cached_threshold = joblib.load(MODEL_PATH)
    print(f"[Evaluate] Loaded model. Cached threshold = {cached_threshold:.4f}")

    # ── Embed datasets ────────────────────────────────────────────────────────
    print("\n[Evaluate] Embedding normal logs...")
    X_normal = embed_lines(load_lines(NORMAL_PATH))

    print("\n[Evaluate] Embedding anomaly logs...")
    X_anomaly = embed_lines(load_lines(ANOMALY_PATH))

    X      = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    # ── Scores ────────────────────────────────────────────────────────────────
    raw_scores    = model.decision_function(X)   # higher = more normal
    anomaly_scores = -raw_scores                 # higher = more anomalous

    # ── Optimal threshold from ROC curve ──────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]   # back to raw_scores space
    roc_auc           = auc(fpr, tpr)
    print(f"[Evaluate] Optimal threshold  = {optimal_threshold:.4f}")

    # ── Predictions at optimal threshold (primary) ────────────────────────────
    y_pred = (raw_scores < optimal_threshold).astype(int)

    # ── Classification report ─────────────────────────────────────────────────
    print("\n── Classification Report (Optimal Threshold) ──────")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Plot 1: Confusion Matrix (optimal threshold) ──────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        title=f"Confusion Matrix (threshold={optimal_threshold:.4f})",
        filename=f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    print(f"[Evaluate] Saved confusion_matrix.png")

    # ── Plot 2: ROC Curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", zorder=5,
               label=f"Optimal threshold = {optimal_threshold:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — IsolationForest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()
    print(f"[Evaluate] Saved roc_curve.png  (AUC={roc_auc:.3f})")

    # ── Plot 3: Precision-Recall Curve ────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — IsolationForest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_curve.png", dpi=150)
    plt.close()
    print(f"[Evaluate] Saved pr_curve.png  (AUC={pr_auc:.3f})")

    # ── Plot 4: Score Distribution ────────────────────────────────────────────
    normal_raw  = raw_scores[:len(X_normal)]
    anomaly_raw = raw_scores[len(X_normal):]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_raw,  bins=20, alpha=0.6, color="steelblue", label="Normal")
    ax.hist(anomaly_raw, bins=20, alpha=0.6, color="tomato",    label="Anomaly")
    ax.axvline(optimal_threshold, color="red",   linestyle="--", lw=1.5,
               label=f"Optimal threshold = {optimal_threshold:.4f}")
    ax.set_xlabel("decision_function score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Normal vs Anomaly")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=150)
    plt.close()
    print(f"[Evaluate] Saved score_distribution.png")

    print(f"\n[Evaluate] All outputs saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()