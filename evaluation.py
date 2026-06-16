"""
evaluate.py — Offline evaluation of the trained AnomalyDetector
Uses cached models/anomaly_model.pkl (must exist — run main.py first to train).

Reads:
    logs/normal_clean.txt  → label 0 (normal)
    logs/anomaly_logs.txt  → label 1 (anomaly)

Outputs:
    - Classification report (Precision, Recall, F1)
    - Confusion matrix
    - ROC curve + AUC
    - Precision-Recall curve + AUC
    - Score distribution plot
    Saved to: evaluation/
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

MODEL_PATH      = "models/anomaly_model.pkl"
NORMAL_PATH     = "logs/normal_logs.txt"
ANOMALY_PATH    = "logs/anomaly_logs.txt"
OUTPUT_DIR      = "evaluation"

# Same regex as query_engine.py
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
        msg = _extract_message(line)
        emb = get_embedding(msg)
        embeddings.append(emb)
        if (i + 1) % 20 == 0:
            print(f"  Embedded {i+1}/{len(lines)}")
    X = np.array(embeddings, dtype=np.float32)
    return normalize(X)


def main():
    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"[Evaluate] No cached model found at {MODEL_PATH}.")
        print("  Run main.py first to train and cache the model.")
        return

    model, threshold = joblib.load(MODEL_PATH)
    print(f"[Evaluate] Loaded model. Threshold={threshold:.4f}")

    # ── Load and embed both datasets ──────────────────────────────────────────
    print("\n[Evaluate] Embedding normal logs...")
    normal_lines = load_lines(NORMAL_PATH)
    X_normal = embed_lines(normal_lines)

    print("\n[Evaluate] Embedding anomaly logs...")
    anomaly_lines = load_lines(ANOMALY_PATH)
    X_anomaly = embed_lines(anomaly_lines)

    # ── Build combined dataset with ground truth labels ───────────────────────
    X = np.vstack([X_normal, X_anomaly])
    # IsolationForest convention: 1=normal, -1=anomaly
    # For sklearn metrics we use: 0=normal, 1=anomaly
    y_true = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    # ── Score every sample ────────────────────────────────────────────────────
    raw_scores = model.decision_function(X)   # higher = more normal
    # Negate so higher score = more anomalous (standard convention for metrics)
    anomaly_scores = -raw_scores

    # Binary predictions at trained threshold
    y_pred = (raw_scores < threshold).astype(int)

    # ── Print classification report ───────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Plot 1: Confusion Matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[Evaluate] Saved confusion_matrix.png")

    # ── Plot 2: ROC Curve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
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
    normal_scores  = raw_scores[:len(X_normal)]
    anomaly_scores_raw = raw_scores[len(X_normal):]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_scores,       bins=20, alpha=0.6, color="steelblue",  label="Normal")
    ax.hist(anomaly_scores_raw,  bins=20, alpha=0.6, color="tomato",     label="Anomaly")
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold={threshold:.3f}")
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