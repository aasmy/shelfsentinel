
import json
import argparse

# from shelfsentinel.perception.zones import load_zones, point_in_polygon #use this import when running the tests
from zones import load_zones, point_in_polygon


def detect_hand_reach_shelf(person, shelf_polygon, person_states, timestamp):
    """
    Detect when either wrist of a tracked person enters the shelf zone.

    Expected person format:
    {
        "track_id": 1,
        "keypoints": {
            "l_wrist": [269.1, 250.4],
            "r_wrist": [281.3, 237.0]
        }
    }

    Returns an event dictionary if a new shelf reach is detected.
    Otherwise returns None.
    """

    track_id = person["track_id"]
    keypoints = person.get("keypoints", {})

    # Get available wrist coordinates.
    left_wrist = keypoints.get("l_wrist")
    right_wrist = keypoints.get("r_wrist")

    wrists = [left_wrist, right_wrist]

    # Check whether at least one visible wrist is inside the shelf.
    wrist_inside = any(
        wrist is not None
        and len(wrist) >= 2
        and point_in_polygon(wrist, shelf_polygon)
        for wrist in wrists
    )

    # If both wrists are missing, we cannot determine whether
    # the person entered or left the shelf zone.
    if left_wrist is None and right_wrist is None:
        return None

    # Get this person's previous state.
    was_inside = person_states.get(track_id, False)

    # A new event occurs when transitioning from outside to inside.
    if wrist_inside and not was_inside:

        person_states[track_id] = True

        return {
            "track_id": track_id,
            "event": "hand_reach_shelf",
            "timestamp": timestamp
        }

    # If visible wrists are outside, reset the state.
    if not wrist_inside:
        person_states[track_id] = False

    return None


def process_video(people_path, zones_path, output_path):
    """
    Read YOLO person detections and generate events.
    """

    zones = load_zones(zones_path)

    if "shelf" not in zones:
        raise ValueError("No shelf zone found in zones.yaml")

    shelf_polygon = zones["shelf"]

    # Tracks whether each person is currently inside the shelf zone.
    person_states = {}

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

                event = detect_hand_reach_shelf(
                    person,
                    shelf_polygon,
                    person_states,
                    timestamp
                )

                if event is not None:
                    events.append(event)

                    print(
                        f"[{timestamp:.3f}s] "
                        f"Person {event['track_id']}: "
                        f"{event['event']}"
                    )

    # Write events as JSONL.
    with open(output_path, "w") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")

    print(f"\nFinished processing.")
    print(f"Events detected: {len(events)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect ShelfSentinel events from YOLO output."
    )

    parser.add_argument(
        "people",
        help="Path to people.jsonl"
    )

    parser.add_argument(
        "--zones",
        default="zones.yaml",
        help="Path to zones.yaml"
    )

    parser.add_argument(
        "--output",
        default="events.jsonl",
        help="Output events JSONL file"
    )

    args = parser.parse_args()

    process_video(
        args.people,
        args.zones,
        args.output
    )
