"""
Layer 2 — Log Parser
Receives a raw log line from Layer 1 (via callback).
Uses regex to extract structured fields.
Assigns severity level via keyword lookup.
Passes parsed dict to Layer 3 (via callback).
"""

import re

# syslog format ISO 8601 Format:
SYSLOG_PATTERN = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.\d+:]*)\s+'  # ISO format
    r'(?P<hostname>\S+)\s+'
    r'(?P<process>[\w/\-\.]+)'
    r'(?:\[(?P<pid>\d+)\])?'
    r':\s+(?P<message>.+)$'
)

# Keyword → level mapping (order matters — check ERROR before INFO)
LEVEL_KEYWORDS = [
    ("CRITICAL", ["critical", "emergency", "panic"]),
    ("ERROR",    ["error", "failed", "failure", "fatal"]),
    ("WARNING",  ["warn", "warning"]),
]

def _assign_level(message: str) -> str:
    msg_lower = message.lower()
    for level, keywords in LEVEL_KEYWORDS:
        if any(kw in msg_lower for kw in keywords):
            return level
    return "INFO"


def parse_line(raw_line: str, callback) -> None:
    """
    Parses a single raw log line.
    On success : calls callback(parsed_dict)
    On failure : prints warning, skips line — never crashes
    """
    match = SYSLOG_PATTERN.match(raw_line.strip())

    if not match:
        print(f"[Parser] Could not parse line, skipping: {raw_line[:80]}")
        return

    parsed = {
        "timestamp": match.group("timestamp"),
        "hostname":  match.group("hostname"),
        "process":   match.group("process"),
        "pid":       match.group("pid") or "",   # empty string if absent
        "message":   match.group("message"),
        "level":     _assign_level(match.group("message")),
    }

    callback(parsed)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Jun 10 10:23:01 ubuntu sshd[1234]: Failed password for root from 192.168.1.1",
        "Jun 10 10:24:00 ubuntu kernel: Out of memory: Kill process 888",
        "Jun 10 10:25:10 ubuntu cron[999]: Job started successfully",
        "THIS IS A MALFORMED LINE WITH NO STRUCTURE",
    ]

    def mock_layer3(parsed: dict):
        print(f"[Layer3 received] → {parsed}\n")

    for line in samples:
        parse_line(line, callback=mock_layer3)
