"""SQLite case-storage and agent workflow integration tests."""

import json
import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelfsentinel.storage.db import (
    connect,
    delete_case,
    list_cases,
    save_case,
    set_review,
    update_case_video_clips,
)
from shelfsentinel.dashboard.data import (
    VideoCleanupError,
    delete_case as delete_dashboard_case,
)
from shelfsentinel.video import clips as video_clips_module


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
    video_clips = [
        {
            "event_type": "exit",
            "track_id": 7,
            "start_time": 1.0,
            "end_time": 4.0,
            "path": "runs/case_1/exit_1.mp4",
        }
    ]
    try:
        with conn:
            case_id = save_case(conn, sample_case(video_clips=video_clips))

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
    assert stored["video_clips"] == video_clips
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


def test_deletes_existing_case_and_preserves_other_cases_and_database(tmp_path):
    conn = connect(tmp_path / "cases.db")
    try:
        with conn:
            deleted_id = save_case(conn, sample_case(track_id=7))
            remaining_id = save_case(
                conn,
                sample_case(track_id=8, severity="L1"),
            )

        with conn:
            assert delete_case(conn, deleted_id)

        remaining = list_cases(conn)
        assert [case["id"] for case in remaining] == [remaining_id]
        assert remaining[0]["track_id"] == 8
        assert remaining[0]["severity"] == "L1"

        with conn:
            new_id = save_case(conn, sample_case(track_id=9))
        assert new_id > remaining_id
        assert len(list_cases(conn)) == 2
    finally:
        conn.close()


def test_deleting_nonexistent_case_returns_false_without_changing_rows(tmp_path):
    conn = connect(tmp_path / "cases.db")
    try:
        with conn:
            case_id = save_case(conn, sample_case())

        with conn:
            assert not delete_case(conn, case_id + 100)

        assert [case["id"] for case in list_cases(conn)] == [case_id]
    finally:
        conn.close()


def test_saves_lists_and_updates_video_clip_metadata(tmp_path):
    conn = connect(tmp_path / "cases.db")
    clips = [
        {
            "event_type": "checkout",
            "track_id": 7,
            "start_time": 1.0,
            "end_time": 4.0,
            "path": "runs/case_1/checkout_1.mp4",
        }
    ]
    try:
        with conn:
            case_id = save_case(conn, sample_case())
        with conn:
            assert update_case_video_clips(conn, case_id, clips)
        assert list_cases(conn)[0]["video_clips"] == clips
        assert not update_case_video_clips(conn, case_id + 1, clips)
    finally:
        conn.close()


def test_existing_database_without_video_clips_is_migrated(tmp_path):
    database_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(database_path)
    raw.execute(
        """
        CREATE TABLE cases (
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
    )
    raw.execute(
        """INSERT INTO cases (
            created_at, video, track_id, severity, summary,
            checks, evidence, dropped, llm_calls
        ) VALUES ('now', 'video.mp4', 3, 'normal', 'summary', '{}', '[]', '[]', 0)"""
    )
    raw.commit()
    raw.close()

    conn = connect(database_path)
    try:
        cases = list_cases(conn)
        assert len(cases) == 1
        assert cases[0]["video_clips"] == []
        assert "video_clips" in {
            row["name"] for row in conn.execute("PRAGMA table_info(cases)")
        }
    finally:
        conn.close()


def test_dashboard_case_deletion_removes_only_its_video_directory(tmp_path):
    database_path = tmp_path / "cases.db"
    runs_dir = tmp_path / "runs"
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"original")
    conn = connect(database_path)
    try:
        with conn:
            case_id = save_case(conn, sample_case())
            other_id = save_case(conn, sample_case(track_id=8))
    finally:
        conn.close()

    target_dir = runs_dir / f"case_{case_id}"
    other_dir = runs_dir / f"case_{other_id}"
    target_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    (target_dir / "evidence.mp4").write_bytes(b"evidence")
    (other_dir / "evidence.mp4").write_bytes(b"other")

    assert delete_dashboard_case(database_path, case_id, runs_dir=runs_dir)
    assert not target_dir.exists()
    assert (other_dir / "evidence.mp4").is_file()
    assert source_video.read_bytes() == b"original"

    conn = connect(database_path)
    try:
        assert [case["id"] for case in list_cases(conn)] == [other_id]
    finally:
        conn.close()


def test_dashboard_case_deletion_handles_missing_directory_and_case(tmp_path):
    database_path = tmp_path / "cases.db"
    runs_dir = tmp_path / "runs"
    conn = connect(database_path)
    try:
        with conn:
            case_id = save_case(conn, sample_case())
    finally:
        conn.close()

    assert delete_dashboard_case(database_path, case_id, runs_dir=runs_dir)
    assert not delete_dashboard_case(
        database_path, case_id, runs_dir=runs_dir
    )


def test_dashboard_case_deletion_never_removes_source_inside_case_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "cases.db"
    runs_dir = tmp_path / "runs"
    conn = connect(database_path)
    try:
        with conn:
            case_id = save_case(
                conn,
                sample_case(video=f"runs/case_1/original.mp4"),
            )
            clip = {
                "event_type": "exit",
                "track_id": 7,
                "path": f"runs/case_{case_id}/exit_1.mp4",
            }
            source_metadata = {"path": f"runs/case_{case_id}/original.mp4"}
            update_case_video_clips(
                conn,
                case_id,
                [clip, source_metadata],
                commit=False,
            )
    finally:
        conn.close()

    case_dir = runs_dir / f"case_{case_id}"
    case_dir.mkdir(parents=True)
    source = case_dir / "original.mp4"
    evidence = case_dir / "exit_1.mp4"
    source.write_bytes(b"source")
    evidence.write_bytes(b"clip")

    with pytest.raises(VideoCleanupError, match="source video"):
        delete_dashboard_case(database_path, case_id, runs_dir=runs_dir)

    assert source.read_bytes() == b"source"
    assert not evidence.exists()
    conn = connect(database_path)
    try:
        assert list_cases(conn) == []
    finally:
        conn.close()


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
    expected_clip = {
        "event_type": "exit",
        "track_id": 7,
        "start_time": 6.0,
        "end_time": 10.0,
        "path": "runs/case_1/exit_1.mp4",
    }

    def create_clips(video_path, case_id, track_id, case_events, evidence):
        assert video_path == "sample.mp4"
        assert track_id == 7
        assert any(event["event"] == "person_exited" for event in case_events)
        existing_conn = connect(database_path)
        try:
            assert [case["id"] for case in list_cases(existing_conn)] == [case_id]
        finally:
            existing_conn.close()
        return [expected_clip]

    monkeypatch.setattr(
        video_clips_module,
        "create_case_clips",
        create_clips,
    )
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
    assert output_cases[0]["video_clips"] == [expected_clip]
    assert stored_cases[0]["video_clips"] == [expected_clip]
