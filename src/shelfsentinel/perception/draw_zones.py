
import cv2
import argparse
import numpy as np

from zones import save_zones


# The zones we want to draw, in order
ZONE_NAMES = ["shelf", "counter", "exit"]

# Stores completed polygons
zones = {}

# Points currently being drawn
current_points = []

# Current zone index
current_zone_index = 0

# Current mouse position (for preview)
mouse_position = None


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks when drawing a polygon."""
    global current_points, mouse_position

    mouse_position = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Added point: ({x}, {y})")


def draw_interface(frame):
    """Draw completed zones and the polygon currently being created."""
    display = frame.copy()

    # Draw completed zones
    for zone_name, polygon in zones.items():
        points = [(int(x), int(y)) for x, y in polygon]

        cv2.polylines(
            display,
            [np.array(points, dtype="int32")],
            True,
            (0, 255, 0),
            2
        )

        # Label near the first point
        cv2.putText(
            display,
            zone_name,
            points[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Draw current points and connecting lines
    for point in current_points:
        cv2.circle(display, point, 5, (0, 0, 255), -1)

    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(
                display,
                current_points[i],
                current_points[i + 1],
                (0, 0, 255),
                2
            )

    # Preview line to mouse
    if current_points and mouse_position:
        cv2.line(
            display,
            current_points[-1],
            mouse_position,
            (255, 0, 0),
            1
        )

    # Instructions
    if current_zone_index < len(ZONE_NAMES):
        current_name = ZONE_NAMES[current_zone_index]
        instruction = f"Drawing: {current_name}"
    else:
        instruction = "All zones completed! Press S to save."

    cv2.putText(
        display,
        instruction,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "Click: Add point | ENTER: Finish | BACKSPACE: Undo",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display,
        "R: Reset current zone | S: Save | Q: Quit",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return display


def main(video_path):
    global current_points, current_zone_index

    # Open video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Read first frame
    success, frame = video.read()
    video.release()

    if not success:
        print("Error: Could not read a frame from the video.")
        return

    window_name = "ShelfSentinel Zone Drawing"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nShelfSentinel Zone Drawing")
    print("--------------------------")
    print("Draw each zone by clicking its corners.")
    print("Press ENTER when you have at least 3 points.\n")

    while True:
        display = draw_interface(frame)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # Finish current polygon
        if key in (10, 13):
            if current_zone_index >= len(ZONE_NAMES):
                continue

            if len(current_points) < 3:
                print("A polygon needs at least 3 points.")
                continue

            zone_name = ZONE_NAMES[current_zone_index]

            zones[zone_name] = list(current_points)

            print(f"Completed zone: {zone_name}")

            current_points.clear()
            current_zone_index += 1

        # Undo last point
        elif key == 8:
            if current_points:
                removed = current_points.pop()
                print(f"Removed point: {removed}")

        # Reset current polygon
        elif key == ord("r"):
            current_points.clear()
            print("Current zone reset.")

        # Save zones
        elif key == ord("s"):
            if current_zone_index < len(ZONE_NAMES):
                print("Please finish all three zones before saving.")
                continue

            save_zones(zones, "./configs/zones.yaml")
            print("Zones saved to ./configs/zones.yaml!")
            break

        # Quit without saving
        elif key == ord("q"):
            print("Exiting without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw ShelfSentinel zones on a video frame."
    )

    parser.add_argument(
        "video",
        help="Path to the video file"
    )

    args = parser.parse_args()

    main(args.video)
