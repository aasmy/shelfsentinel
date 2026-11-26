# scoring for the proposal's evaluation: recall on staged thefts, false alarm rate on normal visits,
# llm calls per visit, for the agent and for the one-call baseline. one labelled video = one visit.

import csv

RANK = {"L2": 3, "L1": 2, "inconclusive": 1, "normal": 0}
FLAGGED = {"L1", "L2"}


def load_labels(path):
    """labels csv: video,scenario,actor,theft[,zones,notes]

    theft is yes/no. scenario is normal / long_browse / pick_put_back / concealment (or anything you like).
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            if not r.get("video"):
                continue
            r["theft"] = r.get("theft", "").lower() in ("yes", "y", "true", "1")
            rows.append(r)
    return rows


def video_verdict(cases):
    """one verdict per video: the most serious case wins. no cases = no shelf reach = normal."""
    if not cases:
        return "normal", 0
    worst = max(cases, key=lambda c: RANK.get(c.get("severity"), 0))["severity"]
    return worst, sum(int(c.get("llm_calls", 0) or 0) for c in cases)


def _rate(num, den):
    return round(num / den, 3) if den else None


def score(rows):
    """rows: dicts with theft (bool), verdict (severity), llm_calls"""
    thefts = [r for r in rows if r["theft"]]
    normals = [r for r in rows if not r["theft"]]
    return {
        "visits": len(rows),
        "thefts": len(thefts),
        "normal_visits": len(normals),
        "recall_l2": _rate(sum(r["verdict"] == "L2" for r in thefts), len(thefts)),
        "recall_flagged": _rate(sum(r["verdict"] in FLAGGED for r in thefts), len(thefts)),
        "false_alarm_rate": _rate(sum(r["verdict"] in FLAGGED for r in normals), len(normals)),
        "false_alarm_rate_l2": _rate(sum(r["verdict"] == "L2" for r in normals), len(normals)),
        "inconclusive_rate": _rate(sum(r["verdict"] == "inconclusive" for r in rows), len(rows)),
        "llm_calls_per_visit": _rate(sum(r["llm_calls"] for r in rows), len(rows)),
    }


def score_by(rows, key):
    """same scores split by a label column, e.g. actor (bias audit) or scenario"""
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key) or "?", []).append(r)
    return {name: score(group) for name, group in sorted(groups.items())}
