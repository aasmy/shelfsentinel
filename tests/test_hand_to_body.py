"""hand_to_body with made up people, no video needed"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.perception.events import ShelfEventDetector

SHELF = [[0, 0], [100, 0], [100, 400], [0, 400]]


def person(wrist, track_id=1):
    # torso is 100px: shoulders at y=200, hips at y=300, standing at x=300
    return {
        "track_id": track_id,
        "bbox": [250, 150, 350, 500],
        "keypoints": {
            "l_shoulder": [280, 200], "r_shoulder": [320, 200],
            "l_hip": [285, 300], "r_hip": [315, 300],
            "r_wrist": wrist,
        },
    }


AT_SHELF = [50, 250]
AT_HIP = [325, 305]
HANGING = [330, 380]  # arm down by the side, well below the hip
IN_FRONT = [300, 230]  # holding the item in front of the chest


def run(wrists, fps=10, **kwargs):
    detector = ShelfEventDetector(SHELF, **kwargs)
    events = []
    for i, wrist in enumerate(wrists):
        events += detector.process_frame([person(wrist)], round(i / fps, 2))
    return events


def names(events):
    return [e["event"] for e in events]


def test_hand_to_hip_after_shelf():
    events = run([AT_SHELF] * 5 + [IN_FRONT] * 5 + [AT_HIP] * 5)
    assert names(events).count("hand_to_body") == 1
    body = next(e for e in events if e["event"] == "hand_to_body")
    assert body["llm_call"] is False
    assert 0 < body["since_leave_s"] <= 5


def test_hanging_arm_is_not_hand_to_body():
    assert "hand_to_body" not in names(run([AT_SHELF] * 5 + [HANGING] * 20))


def test_holding_item_in_front_is_not_hand_to_body():
    assert "hand_to_body" not in names(run([AT_SHELF] * 5 + [IN_FRONT] * 20))


def test_hand_at_hip_without_a_shelf_reach_is_ignored():
    assert "hand_to_body" not in names(run([AT_HIP] * 20))


def test_too_long_after_leaving_the_shelf_is_ignored():
    # 6 seconds of walking around, then hand to hip
    events = run([AT_SHELF] * 5 + [HANGING] * 60 + [AT_HIP] * 5, body_window_seconds=5.0)
    assert "hand_to_body" not in names(events)


def test_needs_a_few_frames_in_a_row():
    assert "hand_to_body" not in names(run([AT_SHELF] * 5 + [AT_HIP, HANGING] * 5))


def test_only_once_per_reach():
    events = run([AT_SHELF] * 5 + [AT_HIP] * 5 + [HANGING] * 3 + [AT_HIP] * 5)
    assert names(events).count("hand_to_body") == 1
    events = run(([AT_SHELF] * 5 + [AT_HIP] * 5 + [HANGING] * 40) * 2)
    assert names(events).count("hand_to_body") == 2  # second real reach later on
