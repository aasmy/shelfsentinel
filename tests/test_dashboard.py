"""Streamlit dashboard smoke tests for cases with and without video evidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelfsentinel.storage.db import connect, save_case, update_case_video_clips
from streamlit.testing.v1 import AppTest


def _case(**overrides):
    case = {
        "video": "source.mp4",
        "track_id": 4,
        "severity": "L1",
        "summary": "evidence-backed case",
        "checks": {},
        "evidence": [],
        "dropped": [],
        "llm_calls": 0,
    }
    case.update(overrides)
    return case


def test_dashboard_renders_cases_with_and_without_clips(tmp_path, monkeypatch):
    database_path = tmp_path / "cases.db"
    conn = connect(database_path)
    try:
        with conn:
            clip_case_id = save_case(
                conn,
                _case(),
            )
            update_case_video_clips(
                conn,
                clip_case_id,
                [
                    {
                        "event_type": "checkout",
                        "track_id": 4,
                        "start_time": 1.0,
                        "end_time": 3.0,
                        "path": "runs/case_1/checkout_1.mp4",
                    }
                ],
                commit=False,
            )
            no_clip_case_id = save_case(
                conn,
                _case(track_id=5, video_clips=[]),
            )
    finally:
        conn.close()

    monkeypatch.chdir(tmp_path)
    clip_path = tmp_path / "runs" / "case_1" / "checkout_1.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"synthetic test video")

    app_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "shelfsentinel"
        / "dashboard"
        / "app.py"
    )
    app = AppTest.from_file(str(app_path)).run()
    app.text_input[0].set_value(str(database_path)).run()
    assert not app.exception
    assert any("No video clips are available" in item.value for item in app.markdown)

    app.session_state["selected_case_id"] = clip_case_id  # same as clicking its row in the table
    app.run()
    assert not app.exception
    assert any(tab.label == "Checkout" for tab in app.tabs)
    assert len(app.get("video")) == 1
    assert not any(
        "Video evidence file is missing" in item.value for item in app.warning
    )

    app.session_state["selected_case_id"] = no_clip_case_id
    app.run()
    assert not app.exception
    assert any("No video clips are available" in item.value for item in app.markdown)


def test_review_buttons_save_the_decision(tmp_path, monkeypatch):
    database_path = tmp_path / "cases.db"
    conn = connect(database_path)
    try:
        with conn:
            case_id = save_case(conn, _case(severity="L2"))
    finally:
        conn.close()
    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "src" / "shelfsentinel" / "dashboard" / "app.py"

    app = AppTest.from_file(str(app_path)).run()
    app.text_input[0].set_value(str(database_path)).run()
    assert not app.exception
    assert not app.selectbox  # no dropdowns anymore
    app.button(key=f"review-confirmed-{case_id}").click().run()
    assert not app.exception

    from shelfsentinel.storage.db import list_cases

    conn = connect(database_path)
    try:
        assert list_cases(conn)[0]["review"] == "confirmed"
    finally:
        conn.close()


def test_upload_helpers():
    from shelfsentinel.dashboard.upload import analyze_command, case_line, progress_lines, safe_name

    assert safe_name("../my store clip (1).MOV") == "my_store_clip_1.mov"
    assert safe_name("evil.exe") == "evil.mp4"
    cmd = analyze_command("data/uploads/a.mp4", "yolo", "runs/cases.db", use_llm=False)
    assert cmd[cmd.index("--auto-zones") + 1] == "yolo" and "--no-llm" in cmd
    cmd = analyze_command("data/uploads/a.mp4", "configs/zones.yaml", "runs/cases.db", use_llm=True, zones_only=True)
    assert cmd[cmd.index("--zones") + 1] == "configs/zones.yaml" and "--zones-only" in cmd and "--no-llm" not in cmd
    assert case_line("case 2 (db 7)  track 4   L1           calls=1  item taken") == "Case #7 · L1 · item taken"
    log = "1) zones\n... 30s\nsome noise\ncase 1 (db 3)  track 4  normal  calls=0  ok\ndone\n"
    assert progress_lines(log) == ["1) zones", "case 1 (db 3)  track 4  normal  calls=0  ok", "done"]
