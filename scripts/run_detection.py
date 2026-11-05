# usage: python scripts/run_detection.py --video data/raw/visit_001.mp4
# saves runs/visit_001/people.jsonl, one line per frame
# add --save-video to also get annotated.mp4 with boxes, ids and keypoints drawn

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.perception.detector import PoseDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    # need fps to turn frame number into seconds
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "people.jsonl"

    detector = PoseDetector(model=args.model)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, fps, (width, height))

    with open(out_file, "w") as f:
        for frame, result in enumerate(detector.track(args.video)):
            t = frame / fps
            people = detector.get_people(result, t)
            line = {"frame": frame, "t": round(t, 3), "people": [p.to_dict() for p in people]}
            f.write(json.dumps(line) + "\n")

            if writer:
                img = result.plot()  # yolo draws the boxes, ids and skeleton for us
                cv2.putText(img, f"{t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                writer.write(img)

            if frame % 100 == 0:
                print(f"frame {frame}  ({t:.1f}s)  {len(people)} people")

    if writer:
        writer.release()
    print(f"done, saved to {out_dir}")


if __name__ == "__main__":
    main()
