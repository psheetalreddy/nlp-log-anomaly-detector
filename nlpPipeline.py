"""
Layer 3 — NLP Pipeline
Input  : parsed log dict from Layer 2
Steps  : tokenize + clean → TF-IDF score → sentence embedding
Output : enriched dict passed to Layer 4 via callback
"""

import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Download required NLTK data on first run
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load once at module level — expensive to reload per line
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_stop_words = set(stopwords.words("english"))

# TF-IDF is fitted incrementally over a growing corpus
# We keep a list of all messages seen so far and refit periodically
_tfidf_vectorizer = TfidfVectorizer()
_tfidf_corpus = []        # all cleaned messages seen so far
_tfidf_fitted = False     # whether vectorizer has been fitted at least once

# Refit TF-IDF every N documents (fitting on every line is too expensive)
TFIDF_REFIT_INTERVAL = 50


# ── Step 1 — Tokenization and Cleaning ───────────────────────────────────────

# Patterns to remove before tokenizing
_IP_PATTERN  = re.compile(r'\b\d{1,3}(\.\d{1,3}){3}\b')   # IPv4
_NUM_PATTERN = re.compile(r'\b\d+\b')                       # standalone numbers

def clean_and_tokenize(message: str) -> list[str]:
    """
    Returns a list of meaningful lowercase tokens.
    Removes: IPs, numbers, punctuation, stopwords, short tokens.
    """
    message = _IP_PATTERN.sub(" ", message)
    message = _NUM_PATTERN.sub(" ", message)

    tokens = word_tokenize(message.lower())

    tokens = [
        t for t in tokens
        if t.isalpha()              # remove punctuation tokens
        and t not in _stop_words   # remove stopwords
        and len(t) > 2             # remove very short tokens (artifacts)
    ]
    return tokens


# ── Step 2 — TF-IDF Score ────────────────────────────────────────────────────

def get_tfidf_score(cleaned_message: str) -> float:
    """
    Returns a single float representing the average TF-IDF score
    of tokens in this message against the current corpus.
    Returns 0.0 if corpus is too small to fit yet.
    """
    global _tfidf_fitted

    _tfidf_corpus.append(cleaned_message)

    # Refit only every N documents to keep it efficient
    if len(_tfidf_corpus) % TFIDF_REFIT_INTERVAL == 0:
        _tfidf_vectorizer.fit(_tfidf_corpus)
        _tfidf_fitted = True

    if not _tfidf_fitted:
        return 0.0

    try:
        vector = _tfidf_vectorizer.transform([cleaned_message])
        dense_array = np.asarray(vector.todense())
    
        # Now .mean() is 100% recognized because it's a true NumPy array
        score = float(dense_array.mean())  # average across all token scores
        return round(score, 6)
    except Exception:
        return 0.0


# ── Step 3 — Sentence Embedding ──────────────────────────────────────────────

def get_embedding(message: str) -> list[float]:
    """
    Returns a 384-dimensional embedding vector for the raw message.
    Uses the original message (not cleaned) — model handles context better.
    """
    vector = _embedding_model.encode(message, show_progress_bar=False)
    return vector.tolist()


# ── Pipeline Entry Point ──────────────────────────────────────────────────────

def process(parsed_dict: dict, callback) -> None:
    """
    Runs the full NLP pipeline on a parsed log dict.
    On success : enriches dict and calls callback(enriched_dict)
    On failure : prints warning, skips — never crashes
    """
    try:
        message = parsed_dict.get("message", "")
        if not message:
            print("[NLP] Empty message, skipping.")
            return

        # Step 1
        tokens = clean_and_tokenize(message)
        cleaned_message = " ".join(tokens)

        # Step 2
        tfidf_score = get_tfidf_score(cleaned_message)

        # Step 3
        embedding = get_embedding(message)

        enriched = {
            **parsed_dict,          # carry forward all Layer 2 fields
            "tokens":      tokens,
            "tfidf_score": tfidf_score,
            "embedding":   embedding,
        }

        callback(enriched)

    except Exception as e:
        print(f"[NLP] Error processing message: {e}, skipping.")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_dicts = [
        {
            "timestamp": "Jun 10 10:23:01",
            "hostname":  "ubuntu",
            "process":   "sshd",
            "pid":       "1234",
            "message":   "Failed password for root from 192.168.1.1",
            "level":     "ERROR",
        },
        {
            "timestamp": "Jun 10 10:24:00",
            "hostname":  "ubuntu",
            "process":   "kernel",
            "pid":       "",
            "message":   "Out of memory: Kill process 888",
            "level":     "CRITICAL",
        },
        {
            "timestamp": "Jun 10 10:25:10",
            "hostname":  "ubuntu",
            "process":   "cron",
            "pid":       "999",
            "message":   "Job started successfully",
            "level":     "INFO",
        },
    ]

    def mock_layer4(enriched: dict):
        print(f"[Layer4 received]")
        print(f"  tokens     : {enriched['tokens']}")
        print(f"  tfidf_score: {enriched['tfidf_score']}")
        print(f"  embedding  : {enriched['embedding'][:5]}... (384 dims)")
        print()

    for d in sample_dicts:
        process(d, callback=mock_layer4)
    
    print(_tfidf_corpus)
    print(_tfidf_fitted)

    