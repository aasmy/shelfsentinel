# sqlite history of cases (long term memory in the proposal)

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB = "runs/cases.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    video TEXT,
    track_id INTEGER,
    severity TEXT,          -- L1, L2, inconclusive, normal
    summary TEXT,
    checks TEXT,            -- json: results of each check
    evidence TEXT,          -- json: claims that passed the guardrail, each with its frame times
    dropped TEXT,           -- json: claims the guardrail removed
    llm_calls INTEGER,
    review TEXT DEFAULT 'pending'   -- pending / confirmed / false_alarm, set by the store owner
)
"""


def connect(path=DEFAULT_DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def save_case(conn, case):
    cur = conn.execute(
        "INSERT INTO cases (created_at, video, track_id, severity, summary, checks, evidence, dropped, llm_calls)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            case["video"],
            case["track_id"],
            case["severity"],
            case["summary"],
            json.dumps(case.get("checks", {})),
            json.dumps(case.get("evidence", [])),
            json.dumps(case.get("dropped", [])),
            case.get("llm_calls", 0),
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_cases(conn, severity=None):
    query = "SELECT * FROM cases"
    args = ()
    if severity:
        query += " WHERE severity = ?"
        args = (severity,)
    rows = conn.execute(query + " ORDER BY id DESC", args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("checks", "evidence", "dropped"):
            d[k] = json.loads(d[k]) if d[k] else []
        out.append(d)
    return out


def set_review(conn, case_id, review):
    if review not in ("pending", "confirmed", "false_alarm"):
        raise ValueError(review)
    conn.execute("UPDATE cases SET review = ? WHERE id = ?", (review, case_id))
    conn.commit()
