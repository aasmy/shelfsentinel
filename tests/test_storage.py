"""SQLite case-storage and agent workflow integration tests."""

import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelfsentinel.storage.db import connect, list_cases, save_case, set_review


def sample_case(**overrides):
    case = {
        "video": "data/sample.mp4",
        "track_id": 7,
        "severity": "L2",
        "summary": "item hidden and left without visiting the counter",
        "checks": {
            "clear_view": True,
            "item_hidden": True,
            "item_taken": False,
            "counter_visit": False,
            "exited": True,
            "looks": 1,
        },
        "evidence": [
            {
                "claim": "item_hidden",
                "what": "puts the item into a pocket",
                "frame_time": 12.5,
            }
        ],
        "dropped": [
            {"claim": "item_taken", "reason": "no valid frame cited"}
        ],
        "llm_calls": 1,
    }
    case.update(overrides)
    return case


def test_saves_and_retrieves_case_with_json_fields_round_tripped(tmp_path):
    database_path = tmp_path / "new" / "nested" / "cases.db"
    conn = connect(database_path)
    try:
        with conn:
            case_id = save_case(conn, sample_case())

        cases = list_cases(conn)
    finally:
        conn.close()

    assert len(cases) == 1
    stored = cases[0]
    assert stored["id"] == case_id
    assert stored["video"] == "data/sample.mp4"
    assert stored["track_id"] == 7
    assert stored["severity"] == "L2"
    assert stored["summary"] == sample_case()["summary"]
    assert stored["checks"] == sample_case()["checks"]
    assert stored["evidence"] == sample_case()["evidence"]
    assert stored["dropped"] == sample_case()["dropped"]
    assert stored["llm_calls"] == 1
    assert stored["review"] == "pending"
    assert stored["created_at"]
    assert database_path.is_file()


def test_filters_cases_by_severity(tmp_path):
    conn = connect(tmp_path / "cases.db")
    try:
        with conn:
            save_case(conn, sample_case(severity="L2"))
            save_case(conn, sample_case(track_id=8, severity="L1"))
            save_case(conn, sample_case(track_id=9, severity="normal"))

        assert [case["severity"] for case in list_cases(conn, "L1")] == ["L1"]
        assert [case["severity"] for case in list_cases(conn, "L2")] == ["L2"]
        assert len(list_cases(conn)) == 3
        assert list_cases(conn, "unknown") == []
    finally:
        conn.close()


def test_updates_review_status_and_rejects_invalid_values(tmp_path):
    conn = connect(tmp_path / "cases.db")
    try:
        with conn:
            case_id = save_case(conn, sample_case())
        with conn:
            assert set_review(conn, case_id, "confirmed")
        assert list_cases(conn)[0]["review"] == "confirmed"

        with pytest.raises(ValueError):
            set_review(conn, case_id, "maybe")

        with conn:
            assert set_review(conn, case_id, "false_alarm")
        assert list_cases(conn)[0]["review"] == "false_alarm"
        assert not set_review(conn, -1, "pending")
    finally:
        conn.close()


def test_missing_database_directory_is_created(tmp_path):
    database_path = tmp_path / "missing" / "folder" / "cases.db"

    conn = connect(database_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 0
    finally:
        conn.close()

    assert database_path.parent.is_dir()


def test_run_agent_persists_cases_to_jsonl_and_sqlite_once_per_track(
    tmp_path, monkeypatch
):
    events_path = tmp_path / "events.jsonl"
    events = [
        {
            "track_id": 7,
            "event": "hand_reach_shelf",
            "t": 1.0,
            "llm_call": True,
        },
        {
            "track_id": 7,
            "event": "hand_leaves_shelf",
            "t": 2.0,
            "llm_call": False,
        },
        {
            "track_id": 7,
            "event": "hand_reach_shelf",
            "t": 4.0,
            "llm_call": True,
        },
        {
            "track_id": 7,
            "event": "hand_leaves_shelf",
            "t": 5.0,
            "llm_call": False,
        },
        {
            "track_id": 7,
            "event": "person_exited",
            "t": 9.0,
            "llm_call": False,
        },
    ]
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    output_path = tmp_path / "output" / "agent_cases.jsonl"
    database_path = tmp_path / "database" / "cases.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent.py",
            "--video",
            "sample.mp4",
            "--events",
            str(events_path),
            "--out",
            str(output_path),
            "--db",
            str(database_path),
            "--no-llm",
        ],
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_agent.py"
    runpy.run_path(str(script), run_name="shelfsentinel_run_agent")["main"]()

    output_cases = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    conn = connect(database_path)
    try:
        stored_cases = list_cases(conn)
    finally:
        conn.close()

    assert len(output_cases) == 1
    assert len(stored_cases) == 1
    assert output_cases[0]["track_id"] == stored_cases[0]["track_id"] == 7
    assert output_cases[0]["severity"] == stored_cases[0]["severity"] == "inconclusive"
    assert stored_cases[0]["llm_calls"] == 0
