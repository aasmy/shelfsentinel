# usage: python scripts/run_detection.py --video data/raw/visit_001.mp4
# saves runs/visit_001/people.jsonl, one line per frame

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
    args = parser.parse_args()

    # need fps to turn frame number into seconds
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "people.jsonl"

    detector = PoseDetector(model=args.model)

    with open(out_file, "w") as f:
        for frame, result in enumerate(detector.track(args.video)):
            t = frame / fps
            people = detector.get_people(result, t)
            line = {"frame": frame, "t": round(t, 3), "people": [p.to_dict() for p in people]}
            f.write(json.dumps(line) + "\n")

            if frame % 100 == 0:
                print(f"frame {frame}  ({t:.1f}s)  {len(people)} people")

    print(f"done, saved to {out_file}")


if __name__ == "__main__":
    main()
