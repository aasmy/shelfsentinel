import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shelfsentinel.perception.events import detect_hand_reach_shelf

# Simple rectangular shelf.
shelf = [
    [100, 100],
    [300, 100],
    [300, 250],
    [100, 250]
]

person_states = {}

# Person 1 approaches the shelf.
person = {
    "track_id": 1,
    "keypoints": {
        "l_wrist": [50, 150],
        "r_wrist": [60, 150]
    }
}

assert detect_hand_reach_shelf(
    person, shelf, person_states, 1.0
) is None


# Person 1 reaches into the shelf with their right wrist.
person = {
    "track_id": 1,
    "keypoints": {
        "l_wrist": [50, 150],
        "r_wrist": [150, 150]
    }
}

event = detect_hand_reach_shelf(
    person, shelf, person_states, 1.5
)

assert event is not None
assert event["event"] == "hand_reach_shelf"
assert event["timestamp"] == 1.5


# Person remains in the shelf zone.
# Should NOT generate another event.
person = {
    "track_id": 1,
    "keypoints": {
        "l_wrist": [50, 150],
        "r_wrist": [160, 150]
    }
}

assert detect_hand_reach_shelf(
    person, shelf, person_states, 2.0
) is None


# Both wrists disappear temporarily.
# Should preserve previous state.
person = {
    "track_id": 1,
    "keypoints": {}
}

assert detect_hand_reach_shelf(
    person, shelf, person_states, 2.5
) is None


# Person moves outside the shelf.
person = {
    "track_id": 1,
    "keypoints": {
        "l_wrist": [50, 150],
        "r_wrist": [350, 150]
    }
}

assert detect_hand_reach_shelf(
    person, shelf, person_states, 3.0
) is None


# Person reaches into shelf again.
person = {
    "track_id": 1,
    "keypoints": {
        "l_wrist": [50, 150],
        "r_wrist": [170, 150]
    }
}

event = detect_hand_reach_shelf(
    person, shelf, person_states, 3.5
)

assert event is not None
assert event["timestamp"] == 3.5

print("All tests passed!")
