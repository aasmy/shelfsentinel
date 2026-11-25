"""Tests for timestamp-based evidence clip extraction."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shelfsentinel.video.clips import create_case_clips, extract_video_clip


def make_video(path, frame_count=25, fps=5.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV has no usable MP4 writer in this environment")
    for frame_index in range(frame_count):
        frame = np.full((24, 32, 3), frame_index * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_video_clip_writes_frames_and_preserves_dimensions(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    output = tmp_path / "clip.mp4"

    metadata = extract_video_clip(source, 1.0, 2.0, output)

    assert output.is_file()
    assert metadata["frames"] == 5
    assert metadata["width"] == 32
    assert metadata["height"] == 24
    captured = cv2.VideoCapture(str(output))
    try:
        assert captured.isOpened()
        assert int(captured.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    finally:
        captured.release()


def test_extract_video_clip_clamps_to_source_boundaries(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    output = tmp_path / "clamped.mp4"

    metadata = extract_video_clip(source, -2.0, 9.0, output)

    assert metadata["actual_start_time"] == 0.0
    assert metadata["actual_end_time"] == 5.0
    assert metadata["frames"] == 25


@pytest.mark.parametrize(
    ("start", "end"),
    [(1.0, 1.0), (2.0, 1.0), (float("nan"), 2.0), (0.0, float("inf"))],
)
def test_extract_video_clip_rejects_invalid_timestamps(tmp_path, start, end):
    source = make_video(tmp_path / "source.mp4")

    with pytest.raises(ValueError):
        extract_video_clip(source, start, end, tmp_path / "invalid.mp4")


def test_extract_video_clip_reports_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_video_clip(
            tmp_path / "missing.mp4",
            0.0,
            1.0,
            tmp_path / "output.mp4",
        )


def test_extract_video_clip_reports_failed_writer_and_leaves_no_partial_file(
    tmp_path, monkeypatch
):
    source = make_video(tmp_path / "source.mp4")

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: ClosedWriter())
    output = tmp_path / "failed.mp4"
    with pytest.raises(OSError, match="writer"):
        extract_video_clip(source, 0.0, 1.0, output)
    assert not output.exists()


def test_case_clips_use_accepted_evidence_and_same_track_event_pairs(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    runs_dir = tmp_path / "runs"
    events = [
        {"track_id": 7, "event": "hand_reach_shelf", "t": 1.0, "llm_call": True},
        {"track_id": 7, "event": "hand_leaves_shelf", "t": 2.0, "llm_call": False},
        {"track_id": 7, "event": "at_counter", "t": 2.5, "llm_call": False},
        {"track_id": 7, "event": "leave_counter", "t": 3.5, "llm_call": False},
        {"track_id": 7, "event": "person_exited", "t": 4.0, "llm_call": False},
        {"track_id": 8, "event": "hand_reach_shelf", "t": 1.0, "llm_call": True},
        {"track_id": 8, "event": "hand_leaves_shelf", "t": 2.0, "llm_call": False},
        {"track_id": 8, "event": "leave_counter", "t": 3.5, "llm_call": False},
    ]
    evidence = [
        {"claim": "item_taken", "frame_time": 1.5},
        {"claim": "item_taken", "frame_time": 1.5},
    ]

    clips = create_case_clips(
        source,
        12,
        7,
        events,
        evidence,
        runs_dir=runs_dir,
    )

    assert [clip["event_type"] for clip in clips] == [
        "shelf_interaction",
        "checkout",
        "exit",
    ]
    assert [clip["start_time"] for clip in clips] == [-1.0, 1.5, 1.0]
    assert [clip["end_time"] for clip in clips] == [5.0, 4.5, 5.0]
    assert all(clip["track_id"] == 7 for clip in clips)
    assert all((runs_dir / f"case_12" / Path(clip["path"]).name).is_file() for clip in clips)
    assert len({clip["path"] for clip in clips}) == len(clips)


def test_case_clips_skip_unmatched_shelf_and_checkout_events(tmp_path):
    source = make_video(tmp_path / "source.mp4")
    events = [
        {"track_id": 1, "event": "hand_reach_shelf", "t": 1.0, "llm_call": True},
        {"track_id": 1, "event": "at_counter", "t": 2.0, "llm_call": False},
        {"track_id": 2, "event": "leave_counter", "t": 3.0, "llm_call": False},
    ]

    clips = create_case_clips(
        source,
        1,
        1,
        events,
        [{"claim": "item_taken", "frame_time": 1.0}],
        runs_dir=tmp_path / "runs",
    )

    assert clips == []
