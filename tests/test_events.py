import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shelfsentinel.perception.events import ShelfEventDetector


shelf = [[100, 100], [300, 100], [300, 250], [100, 250]]
counter = [[400, 100], [600, 100], [600, 250], [400, 250]]
exit_zone = [[700, 100], [900, 100], [900, 250], [700, 250]]


def person(track_id=1, bbox=None, wrist=None):
    keypoints = {}
    if wrist is not None:
        keypoints["r_wrist"] = wrist
    return {
        "track_id": track_id,
        "bbox": bbox,
        "keypoints": keypoints,
    }


def events_named(events, name):
    return [event for event in events if event["event"] == name]


# Existing shelf regression: three consecutive inside frames produce one event.
detector = ShelfEventDetector(shelf)
assert detector.update(
    person(bbox=[0, 0, 100, 100], wrist=[150, 150]), 1.0
) == []
assert detector.update(
    person(bbox=[0, 0, 100, 100], wrist=[160, 150]), 1.1
) == []
event = detector.update(
    person(bbox=[0, 0, 100, 100], wrist=[170, 150]), 1.2
)[0]
assert event == {
    "track_id": 1,
    "event": "hand_reach_shelf",
    "timestamp": 1.2,
    "llm_call": True,
}
assert detector.update(
    person(bbox=[0, 0, 100, 100], wrist=[170, 150]), 1.3
) == []


# Counter: less than two seconds produces no event.
counter_detector = ShelfEventDetector(
    shelf,
    counter_polygon=counter,
)
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 10.0
) == []
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 11.9
) == []

# Counter: 2 seconds confirms once, and staying in the zone does not repeat.
confirmed = counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 12.0
)
at_counter = events_named(confirmed, "at_counter")
assert len(at_counter) == 1
assert at_counter[0]["timestamp"] == 12.0
assert at_counter[0]["llm_call"] is False
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 13.0
) == []

# Counter: a brief boundary fluctuation does not close the checkout visit.
assert counter_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 14.0
) == []
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 15.0
) == []
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 17.0
) == []

# A sustained departure emits one leave event; a later visit can be reported.
assert counter_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 18.0
) == []
leave_events = counter_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 18.8
)
assert len(events_named(leave_events, "leave_counter")) == 1
assert events_named(leave_events, "leave_counter")[0]["track_id"] == 1
assert events_named(leave_events, "leave_counter")[0]["timestamp"] == 18.0
assert counter_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 19.0
) == []
assert counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 20.0
) == []
confirmed = counter_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 22.0
)
assert len(events_named(confirmed, "at_counter")) == 1

# Leaving without a confirmed at_counter visit never emits leave_counter.
unconfirmed_detector = ShelfEventDetector(shelf, counter_polygon=counter)
unconfirmed_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 30.0
)
unconfirmed_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 30.2
)
unconfirmed_leave = unconfirmed_detector.process_frame(
    [person(bbox=[0, 100, 100, 200])], 31.0
)
assert events_named(unconfirmed_leave, "leave_counter") == []

# Counter: a brief visit never produces an event.
brief_detector = ShelfEventDetector(shelf, counter_polygon=counter)
brief_detector.process_frame([person(bbox=[400, 100, 500, 200])], 20.0)
brief_detector.process_frame([person(bbox=[0, 100, 100, 200])], 20.5)
assert brief_detector.process_frame(
    [person(bbox=[400, 100, 500, 200])], 21.0
) == []


# Exit: never entering the exit zone does not create an exit.
exit_detector = ShelfEventDetector(shelf, exit_polygon=exit_zone)
exit_detector.process_frame([person(track_id=1, bbox=[0, 100, 100, 200])], 1.0)
assert exit_detector.process_frame([], 3.0) == []

# Exit: enter, disappear for 1.9 seconds, and not yet confirm.
exit_detector = ShelfEventDetector(shelf, exit_polygon=exit_zone)
exit_detector.process_frame([person(track_id=1, bbox=[700, 100, 800, 200])], 10.0)
assert exit_detector.process_frame([], 11.9) == []

# Exit: disappear to 2 seconds and confirm once.
exit_events = exit_detector.process_frame([], 13.9)
person_exited = events_named(exit_events, "person_exited")
assert len(person_exited) == 1
assert person_exited[0]["track_id"] == 1
assert person_exited[0]["timestamp"] == 13.9
assert person_exited[0]["llm_call"] is False
assert exit_detector.process_frame([], 14.0) == []

# Exit: reappearance outside or before the threshold cancels pending exit.
reappear_detector = ShelfEventDetector(shelf, exit_polygon=exit_zone)
reappear_detector.process_frame(
    [person(track_id=2, bbox=[700, 100, 800, 200])], 20.0
)
reappear_detector.process_frame([], 21.0)
cancelled = reappear_detector.process_frame(
    [person(track_id=2, bbox=[700, 100, 800, 200])], 21.5
)
assert events_named(cancelled, "person_exited") == []
assert reappear_detector.process_frame([], 22.0) == []

# Exit: reappearing outside also cancels exit intent.
outside_reappear = ShelfEventDetector(shelf, exit_polygon=exit_zone)
outside_reappear.process_frame(
    [person(track_id=3, bbox=[700, 100, 800, 200])], 30.0
)
outside_reappear.process_frame([], 31.0)
assert outside_reappear.process_frame(
    [person(track_id=3, bbox=[0, 100, 100, 200])], 31.5
) == []

# Multiple people are tracked independently.
multi_detector = ShelfEventDetector(shelf, exit_polygon=exit_zone)
multi_detector.process_frame([
    person(track_id=4, bbox=[700, 100, 800, 200]),
], 40.0)
multi_detector.process_frame([
    person(track_id=5, bbox=[700, 100, 800, 200]),
], 40.0)
multi_detector.process_frame([], 41.0)
exit_events = multi_detector.process_frame([], 43.0)
assert {event["track_id"] for event in exit_events if event["event"] == "person_exited"} == {4, 5}

# Counter departure state is independent for each track.
multi_counter = ShelfEventDetector(shelf, counter_polygon=counter)
for track_id in (10, 11):
    multi_counter.process_frame(
        [person(track_id=track_id, bbox=[400, 100, 500, 200])], 50.0
    )
    multi_counter.process_frame(
        [person(track_id=track_id, bbox=[400, 100, 500, 200])], 52.0
    )
multi_counter.process_frame(
    [person(track_id=10, bbox=[0, 100, 100, 200])], 53.0
)
multi_counter.process_frame(
    [person(track_id=10, bbox=[0, 100, 100, 200])], 53.8
)
assert multi_counter.counter_states[11]["inside"] is True
assert multi_counter.counter_states[11]["reported"] is True

# Missing zones and missing bounding boxes do not crash.
missing_detector = ShelfEventDetector(shelf)
assert missing_detector.process_frame(
    [person(track_id=6, bbox=None)],
    50.0,
) == []

missing_zones = ShelfEventDetector(shelf, counter_polygon=None, exit_polygon=None)
assert missing_zones.process_frame(
    [person(track_id=7, bbox=[400, 100, 500, 200])],
    51.0,
) == []

print("All tests passed!")
