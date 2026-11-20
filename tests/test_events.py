import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shelfsentinel.perception.events import ShelfEventDetector


shelf = [
    [100, 100],
    [300, 100],
    [300, 250],
    [100, 250]
]


def person(right_wrist):
    keypoints = {} if right_wrist is None else {
        "l_wrist": [50, 150],
        "r_wrist": right_wrist,
    }
    return {"track_id": 1, "keypoints": keypoints}


detector = ShelfEventDetector(shelf)

# Outside to inside starts an in-progress reach, but it is not logged until
# the wrist has remained inside for three consecutive frames.
assert detector.update(person([50, 150]), 1.0) == []
assert detector.update(person([150, 150]), 1.1) == []
assert detector.update(person([160, 150]), 1.2) == []
event = detector.update(person([170, 150]), 1.3)[0]
assert event["event"] == "hand_reach_shelf"
assert event["timestamp"] == 1.3
assert event["llm_call"] is True

# Staying inside does not create duplicate reaches.
assert detector.update(person([170, 150]), 1.4) == []

# Leaving the shelf creates a false event and requires no LLM call.
leave_events = detector.update(person([350, 150]), 1.5)
assert leave_events == [{
    "track_id": 1,
    "event": "hand_leaves_shelf",
    "timestamp": 1.5,
    "llm_call": False,
}]

# A short reach is logged only when it leaves, with llm_call false.
detector.update(person([150, 150]), 2.0)
assert detector.update(person([150, 150]), 2.1) == []
short_events = detector.update(person([350, 150]), 2.2)
assert short_events[0]["event"] == "hand_reach_shelf"
assert short_events[0]["timestamp"] == 2.0
assert short_events[0]["llm_call"] is False
assert short_events[1]["event"] == "hand_leaves_shelf"
assert short_events[1]["llm_call"] is False

# A second sustained reach within three seconds is detected but suppressed.
detector.update(person([150, 150]), 3.0)
detector.update(person([150, 150]), 3.1)
suppressed_events = detector.update(person([150, 150]), 3.2)
assert suppressed_events[0]["event"] == "hand_reach_shelf"
assert suppressed_events[0]["llm_call"] is False

# Missing wrists are inconclusive and do not make a reach event.
missing_detector = ShelfEventDetector(shelf)
missing_detector.update(person([150, 150]), 10.0)
assert missing_detector.update(person(None), 10.1) == []
assert missing_detector.update(person([150, 150]), 10.2) == []
assert missing_detector.update(person([150, 150]), 10.3) == []

print("All tests passed!")
