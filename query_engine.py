"""
Layer 5 — Query Engine + Anomaly Detection
Inputs : enriched log dict from Layer 4 (live stream)
         user queries from CLI (keyword / semantic / anomaly / stats)
Does   : keyword search via inverted index
         semantic search via cosine similarity on embeddings
         anomaly detection via Isolation Forest
Output : colored terminal output
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize
import sqlite3
import os
import joblib

from nlp_pipeline import get_embedding


# ── ANSI colors ───────────────────────────────────────────────────────────────
class Color:
    RESET   = "\033[0m"
    WHITE   = "\033[97m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    BOLD    = "\033[1m"

LEVEL_COLOR = {
    "INFO":     Color.WHITE,
    "WARNING":  Color.YELLOW,
    "ERROR":    Color.RED,
    "CRITICAL": Color.RED + Color.BOLD,
}

MODEL_PATH = "models/anomaly_model.pkl"

def _colorize(log: dict) -> str:
    color = LEVEL_COLOR.get(log.get("level", "INFO"), Color.WHITE)
    return (
        f"{color}[{log.get('level','INFO')}] "
        f"{log.get('timestamp','')} "
        f"{log.get('process','')} "
        f": {log.get('message','')}"
        f"{Color.RESET}"
    )


# ── Anomaly Detector ──────────────────────────────────────────────────────────

import re

# ISO 8601 syslog format used by normal_clean.txt and live syslog
_TRAIN_LINE_PATTERN = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\S]*\s+\S+\s+[\w/\-\.]+(?:\[\d+\])?:\s+(?P<message>.+)$'
)

def _extract_training_message(line: str) -> str:
    """Extracts message field from ISO 8601 syslog line. Falls back to full line."""
    m = _TRAIN_LINE_PATTERN.match(line.strip())
    return m.group("message") if m else line.strip()


class AnomalyDetector:
    def __init__(self, nlp_process_fn):
        self._model = None
        self._nlp_process = nlp_process_fn
        self._anomaly_ids = []
        self._scores = []
        self._threshold = None

    def train(self, log_lines: list[str]) -> None:
        if os.path.isfile(MODEL_PATH):
            self._model, self._threshold = joblib.load(MODEL_PATH)
            print("[AnomalyDetector] Loaded cached model.")
            return

        # Filter out comment lines (anomaly_logs.txt has # category headers)
        log_lines = [l for l in log_lines if not l.startswith("#")]

        print(f"[AnomalyDetector] Generating embeddings for {len(log_lines)} training lines...")
        embeddings = []
        for line in log_lines:
            msg = _extract_training_message(line)
            emb = get_embedding(msg)
            embeddings.append(emb)

        X = np.array(embeddings, dtype=np.float32)
        X = normalize(X)

        self._model = IsolationForest(contamination=0.05, random_state=42)
        self._model.fit(X)

        train_scores = self._model.decision_function(X)
        self._threshold = float(np.percentile(train_scores, 5))

        os.makedirs("models", exist_ok=True)
        joblib.dump((self._model, self._threshold), MODEL_PATH)
        print(f"[AnomalyDetector] Training complete. Threshold={self._threshold:.4f}. Model saved.")

    def predict(self, log_id: int, embedding: list[float]) -> bool:
        if self._model is None or self._threshold is None:
            return False

        x = np.array(embedding, dtype=np.float32).reshape(1, -1)
        x = normalize(x)
        score = float(self._model.decision_function(x)[0])
        self._scores.append((log_id, score))

        is_anomaly = score < self._threshold
        if is_anomaly:
            self._anomaly_ids.append(log_id)
        return is_anomaly

    def get_anomaly_ids(self) -> list[int]:
        return list(self._anomaly_ids)

    def get_metrics(self) -> dict:
        total = len(self._scores)
        anomalies = len(self._anomaly_ids)
        scores_only = [s for _, s in self._scores]
        return {
            "total_scored": total,
            "anomalies_flagged": anomalies,
            "anomaly_rate": round(anomalies / total, 4) if total else 0.0,
            "threshold": round(self._threshold, 4) if self._threshold is not None else None,
            "min_score": round(min(scores_only), 4) if scores_only else None,
            "max_score": round(max(scores_only), 4) if scores_only else None,
            "mean_score": round(float(np.mean(scores_only)), 4) if scores_only else None,
        }

# ── Query Engine ──────────────────────────────────────────────────────────────

class QueryEngine:
    """
    Handles all user queries against the inverted index and SQLite.
    """

    def __init__(self, conn: sqlite3.Connection, index, detector: AnomalyDetector):
        self._conn = conn
        self._index = index
        self._detector = detector

    def _fetch_by_ids(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, timestamp, process, level, message, embedding FROM logs WHERE id IN ({placeholders})",
            ids
        ).fetchall()
        return [
            {
                "id": r[0], "timestamp": r[1], "process": r[2],
                "level": r[3], "message": r[4], "embedding": r[5]
            }
            for r in rows
        ]

    def keyword_search(self, query: str, mode: str = "AND") -> None:
        """
        mode: AND | OR
        Usage: search failed password
               search OR failed memory
        """
        tokens = query.lower().split()
        if mode == "AND":
            ids = self._index.search_and(tokens)
        else:
            ids = self._index.search_or(tokens)

        logs = self._fetch_by_ids(ids)
        if not logs:
            print(f"{Color.CYAN}No results found.{Color.RESET}")
            return

        print(f"{Color.CYAN}Found {len(logs)} result(s):{Color.RESET}")
        for log in logs:
            print(_colorize(log))

    def semantic_search(self, query: str, top_k: int = 5) -> None:
        """
        Converts query to embedding, computes cosine similarity
        against all stored embeddings, returns top K.
        """
        from nlp_pipeline import get_embedding
        from storage import blob_to_embedding

        query_emb = np.array(get_embedding(query), dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        rows = self._conn.execute(
            "SELECT id, timestamp, process, level, message, embedding FROM logs"
        ).fetchall()

        if not rows:
            print(f"{Color.CYAN}No logs in database yet.{Color.RESET}")
            return

        scores = []
        for row in rows:
            emb = np.array(blob_to_embedding(row[5]), dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            similarity = float(np.dot(query_emb, emb))
            scores.append((similarity, row))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        print(f"{Color.CYAN}Top {top_k} semantic matches:{Color.RESET}")
        for sim, row in top:
            log = {"id": row[0], "timestamp": row[1], "process": row[2],
                   "level": row[3], "message": row[4]}
            print(f"  [{sim:.3f}] {_colorize(log)}")

    def show_anomalies(self) -> None:
        ids = self._detector.get_anomaly_ids()
        if not ids:
            print(f"{Color.GREEN}No anomalies detected yet.{Color.RESET}")
            return
        logs = self._fetch_by_ids(ids)
        print(f"{Color.RED}Anomalies detected ({len(logs)}):{Color.RESET}")
        for log in logs:
            print(_colorize(log))

    def show_stats(self) -> None:
        total = self._conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        by_level = self._conn.execute(
            "SELECT level, COUNT(*) FROM logs GROUP BY level"
        ).fetchall()
        anomaly_count = len(self._detector.get_anomaly_ids())
        idx_stats = self._index.stats()

        print(f"{Color.CYAN}── Stats ──────────────────────{Color.RESET}")
        print(f"  Total logs     : {total}")
        print(f"  Anomalies      : {anomaly_count}")
        print(f"  Unique tokens  : {idx_stats['unique_tokens']}")
        print(f"  Index postings : {idx_stats['total_postings']}")
        for level, count in by_level:
            color = LEVEL_COLOR.get(level, Color.WHITE)
            print(f"  {color}{level:<10}{Color.RESET}: {count}")


# ── CLI Parser ────────────────────────────────────────────────────────────────

def handle_command(raw: str, engine: QueryEngine) -> None:
    parts = raw.strip().split(maxsplit=1)
    if not parts:
        return

    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "search":
        if arg.upper().startswith("OR "):
            engine.keyword_search(arg[3:], mode="OR")
        else:
            engine.keyword_search(arg, mode="AND")

    elif cmd == "semantic":
        engine.semantic_search(arg)

    elif cmd == "anomalies":
        engine.show_anomalies()

    elif cmd == "stats":
        engine.show_stats()

    elif cmd == "help":
        print(f"""{Color.CYAN}
    
Commands:
  search <terms>         Keyword AND search
  search OR <terms>      Keyword OR search
  semantic <query>       Semantic similarity search
  anomalies              Show flagged anomaly logs
  stats                  Show index and DB statistics
  exit                   Quit
{Color.RESET}""")

    elif cmd == "exit":
        raise SystemExit

    elif cmd == "metrics":
        m = engine._detector.get_metrics()
        print(f"{Color.CYAN}── Anomaly Detector Metrics ──{Color.RESET}")
        for k, v in m.items():
            print(f"  {k:<20}: {v}")

    else:
        print(f"{Color.YELLOW}Unknown command. Type 'help' for options.{Color.RESET}")