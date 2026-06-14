"""
Layer 4 — Storage
Input  : enriched log dict from Layer 3
Does   : updates inverted index + inserts row into SQLite
Output : passes enriched dict (with assigned id) to Layer 5 via callback
"""

import sqlite3
import threading
import numpy as np

DB_PATH = "logs/logs.db"


# ── SQLite Setup ──────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates the logs table if it doesn't exist.
    Returns a connection object.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hostname  TEXT,
            process   TEXT,
            pid       TEXT,
            level     TEXT,
            message   TEXT,
            tfidf     REAL,
            tokens    TEXT,  
            embedding BLOB
        )
    """)
    conn.commit()
    return conn


# ── Inverted Index ────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    Maps every token to the list of log IDs that contain it.
    Supports AND, OR, NOT boolean queries.

    Structure:
        {
            "failed":   [1, 45, 203],
            "password": [1, 45],
            "memory":   [2, 89]
        }
    """

    def __init__(self):
        self._index = {}      
        self._lock = threading.Lock()

    def add(self, log_id: int, tokens: list[str]) -> None:
        with self._lock:
            for token in tokens:
                if token not in self._index:
                    self._index[token] = []
                if log_id not in self._index[token]:
                    self._index[token].append(log_id)

    def search_and(self, tokens: list[str]) -> list[int]:
        """Returns IDs that contain ALL tokens."""
        with self._lock:
            if not tokens:
                return []
            sets = [set(self._index.get(t, [])) for t in tokens]
            return sorted(set.intersection(*sets))

    def search_or(self, tokens: list[str]) -> list[int]:
        """Returns IDs that contain ANY token."""
        with self._lock:
            if not tokens:
                return []
            sets = [set(self._index.get(t, [])) for t in tokens]
            return sorted(set.union(*sets))

    def search_not(self, include_tokens: list[str], exclude_tokens: list[str]) -> list[int]:
        with self._lock:
            include_sets = [set(self._index.get(t, [])) for t in include_tokens]
            exclude_sets = [set(self._index.get(t, [])) for t in exclude_tokens]
            include = set.union(*include_sets) if include_sets else set()
            exclude = set.union(*exclude_sets) if exclude_sets else set()
            return sorted(include - exclude)

    def stats(self) -> dict:
        with self._lock:
            return {
                "unique_tokens": len(self._index),
                "total_postings": sum(len(v) for v in self._index.values())
            }


# ── Embedding serialization ───────────────────────────────────────────────────

def embedding_to_blob(embedding: list[float]) -> bytes:
    return np.array(embedding, dtype=np.float32).tobytes()

def blob_to_embedding(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


# ── Storage Layer Entry Point ─────────────────────────────────────────────────

class StorageLayer:
    """
    Holds the inverted index and SQLite connection.
    Call process() per enriched log dict from Layer 3.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._conn = init_db(db_path)
        self._index = InvertedIndex()
        self._db_lock = threading.Lock()
        self._repopulate_index()
    
    def _repopulate_index(self) -> None:
        rows = self._conn.execute("SELECT id, tokens FROM logs WHERE tokens IS NOT NULL").fetchall()
        for log_id, tokens_str in rows:
            tokens = [t for t in tokens_str.split(",") if t]
            self._index.add(log_id, tokens)
        if rows:
            print(f"[Storage] Repopulated index with {len(rows)} existing logs.")
    

    def process(self, enriched_dict: dict, callback) -> None:
        try:
            embedding_blob = embedding_to_blob(enriched_dict["embedding"])

            with self._db_lock:
                cursor = self._conn.execute(
                    """
                    INSERT INTO logs (timestamp, hostname, process, pid, level, message, tfidf, tokens, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        enriched_dict["timestamp"],
                        enriched_dict["hostname"],
                        enriched_dict["process"],
                        enriched_dict["pid"],
                        enriched_dict["level"],
                        enriched_dict["message"],
                        enriched_dict["tfidf_score"],
                        ",".join(enriched_dict["tokens"]),
                        embedding_blob,
                    )
                )
                self._conn.commit()
                log_id = cursor.lastrowid

            if log_id is None:
                raise ValueError("Database failed to return a valid row ID. Check schema.")

            # Update inverted index with the assigned DB id
            self._index.add(log_id, enriched_dict["tokens"])

            # Pass enriched dict forward with db id attached
            callback({**enriched_dict, "id": log_id})

        except Exception as e:
            print(f"[Storage] Error storing log: {e}, skipping.")

    def get_index(self) -> InvertedIndex:
        return self._index

    def get_connection(self) -> sqlite3.Connection:
        return self._conn


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    storage = StorageLayer()

    samples = [
        {
            "timestamp": "Jun 10 10:23:01", "hostname": "ubuntu",
            "process": "sshd", "pid": "1234", "level": "ERROR",
            "message": "Failed password for root from 192.168.1.1",
            "tfidf_score": 0.21,
            "tokens": ["failed", "password", "root"],
            "embedding": [0.1] * 384,
        },
        {
            "timestamp": "Jun 10 10:24:00", "hostname": "ubuntu",
            "process": "kernel", "pid": "", "level": "CRITICAL",
            "message": "Out of memory: Kill process 888",
            "tfidf_score": 0.18,
            "tokens": ["memory", "kill", "process"],
            "embedding": [0.2] * 384,
        },
        {
            "timestamp": "Jun 10 10:25:10", "hostname": "ubuntu",
            "process": "cron", "pid": "999", "level": "INFO",
            "message": "Failed job started successfully",
            "tfidf_score": 0.05,
            "tokens": ["memory", "kill", "process"],
            "embedding": [0.3] * 384,
        },
    ]

    def mock_layer5(stored: dict):
        print(f"[Layer5 received] id={stored['id']} level={stored['level']} message={stored['message']}")

    for s in samples:
        storage.process(s, callback=mock_layer5)

    # Test boolean queries
    idx = storage.get_index()
    print(idx._index)
    print(f"\nAND ['failed', 'password'] : {idx.search_and(['failed', 'password'])}")
    print(f"OR  ['failed', 'memory']   : {idx.search_or(['failed', 'memory'])}")
    print(f"NOT include=['failed'] exclude=['password'] : {idx.search_not(['failed'], ['password'])}")
    print(f"\nIndex stats: {idx.stats()}")