
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from shelfsentinel.perception.zones import load_zones, point_in_polygon
except ImportError:
    from zones import load_zones, point_in_polygon


class ShelfEventDetector:
    """Stateful hand reach/leave detector for one or more tracked people."""

    def __init__(
        self,
        shelf_polygon,
        counter_polygon=None,
        exit_polygon=None,
        min_frames=3,
        suppression_seconds=3.0,
        body_window_seconds=5.0,
        body_x_ratio=0.35,
        body_y_ratio=0.25,
    ):
        self.shelf_polygon = shelf_polygon
        self.counter_polygon = counter_polygon
        self.exit_polygon = exit_polygon
        self.min_frames = min_frames
        self.suppression_seconds = suppression_seconds
        self.states = {}
        self.counter_states = {}
        self.exit_states = {}
        self.counter_leave_grace_seconds = 0.75
        # hand_to_body: a wrist at hip height next to the hip (pocket / bag on the hip)
        # shortly after the hand left the shelf. sizes are relative to the torso length
        # so it works for people close to or far from the camera
        self.body_window_seconds = body_window_seconds
        self.body_x_ratio = body_x_ratio
        self.body_y_ratio = body_y_ratio
        self.body_states = {}

    def process_frame(self, people, timestamp):
        """Process shelf, counter, and exit state for one complete frame."""
        events = []
        present_ids = set()

        for person in people:
            if hasattr(person, "to_dict"):
                person = person.to_dict()

            track_id = person["track_id"]
            present_ids.add(track_id)

            shelf_events = self.update(person, timestamp) or []
            events.extend(shelf_events)
            events.extend(self._update_hand_to_body(track_id, person, timestamp, shelf_events))
            events.extend(self._update_counter(track_id, person, timestamp))
            events.extend(self._update_exit_presence(track_id, person, timestamp))

        events.extend(self._update_exit_disappearance(present_ids, timestamp))
        return events

    def end_frame(self, people, timestamp):
        """Compatibility alias for processing all per-frame events."""
        return self.process_frame(people, timestamp)

    @staticmethod
    def person_bottom_center(person):
        """Return the bottom-center point of a person's bounding box."""
        bbox = person.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        x1, _, x2, y2 = bbox
        return [(x1 + x2) / 2, y2]

    def person_in_zone(self, person, zone_polygon):
        """Return whether a person's bottom-center is inside a zone."""
        if zone_polygon is None:
            return False
        point = self.person_bottom_center(person)
        if point is None:
            return False
        return point_in_polygon(point, zone_polygon)

    def _update_counter(self, track_id, person, timestamp):
        if self.counter_polygon is None:
            return []

        inside = self.person_in_zone(person, self.counter_polygon)
        state = self.counter_states.setdefault(track_id, {
            "inside": False,
            "entered_at": None,
            "reported": False,
            "leave_started_at": None,
        })

        if inside:
            if not state["inside"]:
                state["inside"] = True
                state["entered_at"] = timestamp
                state["reported"] = False
                state["leave_started_at"] = None
            elif (
                not state["reported"]
                and timestamp - state["entered_at"] >= 2.0
            ):
                state["reported"] = True
                return [{
                    "track_id": track_id,
                    "event": "at_counter",
                    "timestamp": round(timestamp, 3),
                    "llm_call": False,
                }]
            state["leave_started_at"] = None
        else:
            if not state["inside"]:
                return []
            if state["leave_started_at"] is None:
                state["leave_started_at"] = timestamp
                return []
            if timestamp - state["leave_started_at"] < self.counter_leave_grace_seconds:
                return []

            events = []
            if state["reported"]:
                events.append({
                    "track_id": track_id,
                    "event": "leave_counter",
                    "timestamp": round(state["leave_started_at"], 3),
                    "llm_call": False,
                })
            state["inside"] = False
            state["entered_at"] = None
            state["reported"] = False
            state["leave_started_at"] = None
            return events
        return []

    def _update_exit_presence(self, track_id, person, timestamp):
        if self.exit_polygon is None:
            return []

        inside = self.person_in_zone(person, self.exit_polygon)
        state = self.exit_states.setdefault(track_id, {
            "seen_in_exit": False,
            "disappeared_since": None,
            "exited": False,
        })

        if state["exited"]:
            return []

        if inside:
            state["seen_in_exit"] = True
            state["disappeared_since"] = None
        elif state["seen_in_exit"] and state["disappeared_since"] is not None:
            # The track returned during the grace period outside the exit:
            # cancel the pending exit rather than confirming it.
            state["seen_in_exit"] = False
            state["disappeared_since"] = None
        return []

    def _update_exit_disappearance(self, present_ids, timestamp):
        if self.exit_polygon is None:
            return []

        events = []
        for track_id, state in self.exit_states.items():
            if state["exited"] or not state["seen_in_exit"]:
                continue

            if track_id in present_ids:
                continue

            if state["disappeared_since"] is None:
                state["disappeared_since"] = timestamp
            elif timestamp - state["disappeared_since"] >= 2.0:
                state["exited"] = True
                events.append({
                    "track_id": track_id,
                    "event": "person_exited",
                    "timestamp": round(timestamp, 3),
                    "llm_call": False,
                })
        return events

    @staticmethod
    def wrist_at_hip(keypoints, x_ratio=0.35, y_ratio=0.25):
        """True if a wrist is next to a hip at hip height (hand at a pocket or a bag on the hip).

        arms hanging by the side put the wrist well below the hip, so they dont count.
        needs both a shoulder and a hip to know the torso size, otherwise returns False.
        """
        shoulders = [keypoints.get(k) for k in ("l_shoulder", "r_shoulder") if keypoints.get(k)]
        hips = [keypoints.get(k) for k in ("l_hip", "r_hip") if keypoints.get(k)]
        wrists = [keypoints.get(k) for k in ("l_wrist", "r_wrist") if keypoints.get(k)]
        if not shoulders or not hips or not wrists:
            return False

        shoulder_y = sum(p[1] for p in shoulders) / len(shoulders)
        hip_y = sum(p[1] for p in hips) / len(hips)
        torso = abs(hip_y - shoulder_y)
        if torso < 1:
            return False

        for wx, wy in (w[:2] for w in wrists):
            for hx, hy in (h[:2] for h in hips):
                if abs(wx - hx) <= x_ratio * torso and abs(wy - hy) <= y_ratio * torso:
                    return True
        return False

    def _update_hand_to_body(self, track_id, person, timestamp, shelf_events):
        state = self.body_states.setdefault(track_id, {"last_leave": None, "streak": 0, "reported": True})

        for event in shelf_events:
            if event["event"] == "hand_reach_shelf" and event["llm_call"]:
                state["reported"] = False  # new real reach, can report again
            if event["event"] == "hand_leaves_shelf":
                state["last_leave"] = event["timestamp"]

        hand_in_shelf = self.states.get(track_id, {}).get("inside", False)
        recent = state["last_leave"] is not None and timestamp - state["last_leave"] <= self.body_window_seconds
        at_hip = self.wrist_at_hip(person.get("keypoints", {}), self.body_x_ratio, self.body_y_ratio)

        if state["reported"] or hand_in_shelf or not recent or not at_hip:
            state["streak"] = 0
            return []

        state["streak"] += 1
        if state["streak"] < self.min_frames:
            return []

        state["reported"] = True
        return [{
            "track_id": track_id,
            "event": "hand_to_body",
            "timestamp": timestamp,
            "llm_call": False,
            "since_leave_s": round(timestamp - state["last_leave"], 2),
        }]

    def update(self, person, timestamp):
        """Return a newly emitted event, or a list of events, for this person frame."""
        if hasattr(person, "to_dict"):
            person = person.to_dict()

        track_id = person["track_id"]
        keypoints = person.get("keypoints", {})
        left_wrist = keypoints.get("l_wrist")
        right_wrist = keypoints.get("r_wrist")

        # Missing wrist keypoints are inconclusive, so preserve the current state.
        if left_wrist is None and right_wrist is None:
            if track_id in self.states and self.states[track_id]["inside"]:
                self.states[track_id]["streak"] = 0
            return []

        wrists = [wrist for wrist in (left_wrist, right_wrist) if wrist is not None]
        wrist_inside = any(
            len(wrist) >= 2 and point_in_polygon(wrist, self.shelf_polygon)
            for wrist in wrists
        )

        state = self.states.setdefault(track_id, {
            "inside": False,
            "streak": 0,
            "entered_at": None,
            "reach_reported": False,
            "last_llm_event": None,
        })

        if wrist_inside:
            if not state["inside"]:
                state["inside"] = True
                state["streak"] = 1
                state["entered_at"] = timestamp
                state["reach_reported"] = False
                return []

            state["streak"] += 1
            if state["reach_reported"] or state["streak"] < self.min_frames:
                return []

            state["reach_reported"] = True
            last_llm_event = state["last_llm_event"]
            llm_call = not (
                last_llm_event is not None
                and timestamp - last_llm_event <= self.suppression_seconds
            )
            if llm_call:
                state["last_llm_event"] = timestamp

            return [{
                "track_id": track_id,
                "event": "hand_reach_shelf",
                "timestamp": timestamp,
                "llm_call": llm_call,
            }]

        if state["inside"]:
            events = []
            if not state["reach_reported"]:
                events.append({
                    "track_id": track_id,
                    "event": "hand_reach_shelf",
                    "timestamp": state["entered_at"],
                    "llm_call": False,
                })

            events.append({
                "track_id": track_id,
                "event": "hand_leaves_shelf",
                "timestamp": timestamp,
                "llm_call": False,
            })
            state["inside"] = False
            state["streak"] = 0
            state["entered_at"] = None
            state["reach_reported"] = False
            return events

        return []


def detect_hand_reach_shelf(person, shelf_polygon, person_states, timestamp):
    """Compatibility wrapper around ShelfEventDetector."""
    track_id = person["track_id"]
    state = person_states.setdefault(track_id, {
        "detector": None,
        "inside": False,
        "streak": 0,
        "entered_at": None,
        "reach_reported": False,
        "last_llm_event": None,
    })

    if state["detector"] is None:
        state["detector"] = ShelfEventDetector(shelf_polygon)

    events = state["detector"].update(person, timestamp)
    for event in events:
        if event["event"] == "hand_reach_shelf":
            return event
    return None


def process_video(people_path, zones_path, output_path):
    """
    Read YOLO person detections and generate events.
    """

    zones = load_zones(zones_path)

    if "shelf" not in zones:
        raise ValueError("No shelf zone found in zones.yaml")

    shelf_polygon = zones["shelf"]

    detector = ShelfEventDetector(
        shelf_polygon,
        counter_polygon=zones.get("counter"),
        exit_polygon=zones.get("exit"),
    )
    events = []

    # Read one JSON object per line.
    with open(people_path, "r") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            frame_data = json.loads(line)

            timestamp = frame_data["t"]

            # Process every person detected in this frame.
            people = frame_data.get("people", [])
            for event in detector.process_frame(people, timestamp):
                events.append(event)
                print(
                    f"[{event['timestamp']:.3f}s] "
                    f"Person {event['track_id']}: "
                    f"{event['event']} "
                    f"llm_call={event['llm_call']}"
                )

    # Write events as JSONL.
    with open(output_path, "w") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")

    print(f"\nFinished processing.")
    print(f"Events detected: {len(events)}")
    print(f"Saved to: {output_path}")


def detect_direct_video(video_path, zones_path, output_path):
    """Run YOLO directly and write shelf events as JSONL."""
    import cv2
    from shelfsentinel.perception.detector import PoseDetector

    zones = load_zones(zones_path)
    if "shelf" not in zones:
        raise ValueError("No shelf zone found in zones.yaml")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    detector = ShelfEventDetector(
        zones["shelf"],
        counter_polygon=zones.get("counter"),
        exit_polygon=zones.get("exit"),
    )
    events = []
    pose_detector = PoseDetector()

    for frame, result in enumerate(pose_detector.track(video_path)):
        timestamp = round(frame / fps, 3)
        people = pose_detector.get_people(result, timestamp)
        for event in detector.process_frame(people, timestamp):
            events.append(event)
            print(
                f"[{event['timestamp']:.3f}s] "
                f"Person {event['track_id']}: "
                f"{event['event']} "
                f"llm_call={event['llm_call']}"
            )

    with open(output_path, "w") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")

    print("\nFinished processing.")
    print(f"Events detected: {len(events)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect ShelfSentinel events from YOLO output."
    )

    parser.add_argument(
        "people",
        nargs="?",
        help="Path to people.jsonl"
    )
    parser.add_argument(
        "--video",
        help="Run detection directly on a video instead of reading people.jsonl"
    )

    parser.add_argument(
        "--zones",
        default="./configs/zones.yaml",
        help="Path to zones.yaml"
    )

    parser.add_argument(
        "--output",
        default="events.jsonl",
        help="Output events JSONL file"
    )

    args = parser.parse_args()

    if args.video:
        detect_direct_video(args.video, args.zones, args.output)
    elif args.people:
        process_video(args.people, args.zones, args.output)
    else:
        parser.error("provide a people.jsonl path or --video")
