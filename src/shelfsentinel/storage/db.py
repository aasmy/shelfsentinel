"""SQLite persistence for ShelfSentinel investigation cases."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = "runs/cases.db"
REVIEW_STATUSES = {"pending", "confirmed", "false_alarm"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    video TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    checks TEXT NOT NULL,
    evidence TEXT NOT NULL,
    dropped TEXT NOT NULL,
    llm_calls INTEGER NOT NULL,
    review TEXT NOT NULL DEFAULT 'pending'
)
"""

JSON_FIELDS = {
    "checks": {},
    "evidence": [],
    "dropped": [],
}


def connect(path=DEFAULT_DB):
    """Open the database, creating its parent directory and schema if needed.

    The caller owns the connection and transaction. Use it as a context manager
    to commit writes, and close it when finished.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def save_case(conn, case, commit=True):
    """Insert a generated case and return its database ID.

    By default the insert is committed immediately for standalone callers.
    Batch callers can pass ``commit=False`` and commit once after the batch.
    """
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    values = (
        created_at,
        case["video"],
        case["track_id"],
        case["severity"],
        case["summary"],
        json.dumps(case.get("checks", {}), allow_nan=False),
        json.dumps(case.get("evidence", []), allow_nan=False),
        json.dumps(case.get("dropped", []), allow_nan=False),
        case.get("llm_calls", 0),
    )
    cursor = conn.execute(
        """
        INSERT INTO cases (
            created_at, video, track_id, severity, summary,
            checks, evidence, dropped, llm_calls
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    if commit:
        conn.commit()
    return cursor.lastrowid


def list_cases(conn, severity=None):
    """Return cases newest-first, optionally filtered by severity."""
    if severity is None:
        rows = conn.execute("SELECT * FROM cases ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM cases WHERE severity = ? ORDER BY id DESC",
            (severity,),
        ).fetchall()

    cases = []
    for row in rows:
        case = dict(row)
        for field, default in JSON_FIELDS.items():
            value = case[field]
            case[field] = json.loads(value) if value else default.copy()
        cases.append(case)
    return cases


def set_review(conn, case_id, review, commit=True):
    """Set review status and return whether the case existed.

    By default the change is committed immediately; pass ``commit=False`` to
    include it in a caller-managed transaction.
    """
    if review not in REVIEW_STATUSES:
        raise ValueError(f"review must be one of {sorted(REVIEW_STATUSES)}")
    cursor = conn.execute(
        "UPDATE cases SET review = ? WHERE id = ?",
        (review, case_id),
    )
    if commit:
        conn.commit()
    return cursor.rowcount == 1
