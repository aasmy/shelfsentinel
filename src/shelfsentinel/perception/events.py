
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

    def __init__(self, shelf_polygon, min_frames=3, suppression_seconds=3.0):
        self.shelf_polygon = shelf_polygon
        self.min_frames = min_frames
        self.suppression_seconds = suppression_seconds
        self.states = {}

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

    detector = ShelfEventDetector(shelf_polygon)
    events = []

    # Read one JSON object per line.
    with open(people_path, "r") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            frame_data = json.loads(line)

            timestamp = frame_data["t"]

            # Process every person detected in this frame.
            for person in frame_data.get("people", []):
                for event in detector.update(person, timestamp):
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

    detector = ShelfEventDetector(zones["shelf"])
    events = []
    pose_detector = PoseDetector()

    for frame, result in enumerate(pose_detector.track(video_path)):
        timestamp = round(frame / fps, 3)
        for person in pose_detector.get_people(result, timestamp):
            for event in detector.update(person, timestamp):
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
