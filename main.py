"""
main.py — Entry point
Wires all 5 layers together.

Thread 1 (background) : watches log file → parses → NLP → stores → anomaly check
Thread 2 (foreground) : CLI query loop

Usage:
    python3 main.py                        # watches /var/log/syslog
    python3 main.py logs/sample.log        # watches a custom file
"""

import sys
import time
import threading
import os

from watcher       import LogWatcher
from parser        import parse_line
from nlp_pipeline  import process as nlp_process
from storage       import StorageLayer
from query_engine  import AnomalyDetector, QueryEngine, handle_command, Color

TRAINING_DATA = "logs/normal_logs.txt"
DEFAULT_LOG   = "/var/log/syslog"


# ── Startup: train anomaly detector ──────────────────────────────────────────

def load_training_lines(path: str) -> list[str]:
    if not os.path.isfile(path):
        print(f"[Main] Training file not found: {path}. Anomaly detection disabled.")
        return []
    with open(path, "r", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


# ── Pipeline: called per new raw log line ─────────────────────────────────────

def build_pipeline(storage: StorageLayer, detector: AnomalyDetector):
    """
    Returns a single function: raw_line -> parse -> NLP -> store -> anomaly check
    This is passed to Layer 1 as the callback.
    """
    def on_new_log(raw_line: str):
        # Layer 2
        def after_parse(parsed):
            # Layer 3
            def after_nlp(enriched):
                # Layer 4
                def after_store(stored):
                    # Layer 5 — anomaly check
                    is_anomaly = detector.predict(stored["id"], stored["embedding"])
                    if is_anomaly:
                        print(
                            f"\n{Color.RED}[ANOMALY DETECTED]{Color.RESET} "
                            f"{stored['timestamp']} {stored['process']}: {stored['message']}"
                        )
                storage.process(enriched, callback=after_store)
            nlp_process(parsed, callback=after_nlp)
        parse_line(raw_line, callback=after_parse)

    return on_new_log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG

    # Init storage
    os.makedirs("logs", exist_ok=True)
    storage = StorageLayer()

    # Train anomaly detector
    detector = AnomalyDetector(nlp_process)
    training_lines = load_training_lines(TRAINING_DATA)
    if training_lines:
        detector.train(training_lines)

    # Build query engine
    engine = QueryEngine(
        conn=storage.get_connection(),
        index=storage.get_index(),
        detector=detector
    )

    # Build pipeline callback
    pipeline = build_pipeline(storage, detector)

    # Start Layer 1 in background thread
    try:
        watcher = LogWatcher(log_path, callback=pipeline)
    except FileNotFoundError as e:
        print(f"[Main] {e}")
        sys.exit(1)

    bg_thread = threading.Thread(target=watcher.start, daemon=True)
    bg_thread.start()

    # Foreground: CLI loop
    print(f"\n{Color.GREEN}Log Analyzer running.{Color.RESET} Watching: {log_path}")
    print(f"Type {Color.CYAN}'help'{Color.RESET} for available commands.\n")

    try:
        while True:
            try:
                raw = input(f"{Color.CYAN}> {Color.RESET}")
                handle_command(raw, engine)
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        watcher.stop()
        print("\n[Main] Shutdown complete.")


if __name__ == "__main__":
    main()