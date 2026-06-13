"""
Layer 1 — File Watcher
Monitors a log file for new lines using watchdog (inotify under the hood).
Emits each new raw line to a registered callback (Layer 2 entry point).
"""

import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class LogFileHandler(FileSystemEventHandler):
    """
    Handles filesystem events for the watched log file.
    Reads only newly appended lines on each modification event.
    """

    def __init__(self, filepath: str, callback):
        """
        filepath : absolute path to the log file being watched
        callback : function(line: str) -> None  — called per new raw line
        """
        self._filepath = os.path.abspath(filepath)
        self._callback = callback
        self._lock = threading.Lock()

        # Seek to end on startup — we don't want to replay existing lines
        # Change to 0 if you want to ingest the full file on start
        with open(self._filepath, "r", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            self._position = f.tell()

    def on_modified(self, event):
        # watchdog watches the whole directory; filter to our file only
        if os.path.abspath(event.src_path) != self._filepath:
            return

        with self._lock:
            try:
                with open(self._filepath, "r", errors="replace") as f:
                    f.seek(self._position)
                    new_content = f.read()
                    self._position = f.tell()

                for line in new_content.splitlines():
                    line = line.strip()
                    if line:  # skip blank lines
                        self._callback(line)

            except (OSError, IOError) as e:
                print(f"[Watcher] Error reading file: {e}")


class LogWatcher:
    """
    Public interface for Layer 1.
    Usage:
        watcher = LogWatcher("/var/log/syslog", callback=layer2_parse)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(self, filepath: str, callback):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Log file not found: {filepath}")

        self._filepath = filepath
        self._handler = LogFileHandler(filepath, callback)
        self._observer = Observer()

        # Watch the directory containing the file (watchdog requirement)
        watch_dir = os.path.dirname(os.path.abspath(filepath))
        self._observer.schedule(self._handler, path=watch_dir, recursive=False)

    def start(self):
        self._observer.start()
        print(f"[Watcher] Monitoring: {self._filepath}")

    def stop(self):
        self._observer.stop()
        self._observer.join()
        print("[Watcher] Stopped.")


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    #log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/sample.log"
    log_path = "/var/log/syslog"

    def mock_layer2(raw_line: str):
        """Simulates Layer 2 receiving the line."""
        print(f"[Layer2 received] → {raw_line}")

    watcher = LogWatcher(log_path, callback=mock_layer2)
    watcher.start()

    print("Watcher running. Append lines to the file to test. Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
