# full rough pipeline: video -> detector -> hand on shelf moments -> frames -> gemini -> results
# usage: python scripts/run_pipeline.py --video data/test.mp4
#        python scripts/run_pipeline.py --video data/test.mp4 --no-llm   (just find the moments, no api calls)
# saves runs/<video>/cases.jsonl

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.frames import get_frames
from shelfsentinel.perception.detector import PoseDetector
from shelfsentinel.perception.zones import load_zones, point_in_polygon

PROMPT = (
    "These are {n} frames from a store camera, in order, taken at {times} seconds. "
    "Focus on the person whose hand is at the shelf. "
    "Describe only what their hands do with items (picking up, putting back, putting in pocket or bag). "
    "Do not describe or guess who the person is. "
    'Answer as JSON: {{"hands": "...", "item_taken": true/false, "item_hidden": true/false, "clear_view": true/false}}'
)


def find_shelf_moments(detector, video_path, shelf, fps, min_frames=3, cooldown=3.0):
    # temporary simple version of hand_reach_shelf until events.py is ready
    # a moment = a wrist stays inside the shelf zone for min_frames frames in a row
    streak = {}  # track_id -> frames in a row with a wrist in the shelf
    last = {}  # track_id -> time of their last moment, so we dont spam the same reach
    moments = []

    for frame, result in enumerate(detector.track(video_path)):
        t = frame / fps
        for p in detector.get_people(result, t):
            wrists = [p.keypoints[w] for w in ("l_wrist", "r_wrist") if w in p.keypoints]
            in_shelf = any(point_in_polygon(w, shelf) for w in wrists)
            streak[p.track_id] = streak.get(p.track_id, 0) + 1 if in_shelf else 0

            if streak[p.track_id] == min_frames and t - last.get(p.track_id, -999) > cooldown:
                last[p.track_id] = t
                moments.append({"track_id": p.track_id, "event": "hand_reach_shelf", "t": round(t, 2)})
                print(f"{t:6.1f}s  track {p.track_id}  hand_reach_shelf")

        if frame % 300 == 0:
            print(f"... {t:.0f}s")

    return moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--max-calls", type=int, default=5, help="max gemini calls per run, to save quota")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    zones = load_zones(args.zones)
    if "shelf" not in zones:
        sys.exit(f"no shelf zone in {args.zones}, run draw_zones.py first")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    print("1) finding hand on shelf moments")
    moments = find_shelf_moments(PoseDetector(), args.video, zones["shelf"], fps)
    print(f"found {len(moments)} moments\n")

    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cases.jsonl"

    gemini = None
    if not args.no_llm:
        from shelfsentinel.llm.gemini import Gemini

        gemini = Gemini()

    print("2) asking gemini" if gemini else "2) skipping gemini (--no-llm)")
    with open(out_file, "w") as f:
        for i, m in enumerate(moments):
            case = dict(m)
            frames = get_frames(args.video, m["t"])
            case["frame_times"] = [ft for ft, _ in frames]

            if gemini and i < args.max_calls:
                prompt = PROMPT.format(n=len(frames), times=case["frame_times"])
                try:
                    case["llm"] = gemini.ask_with_images(prompt, [fr for _, fr in frames])
                except Exception as e:  # keep going if one call fails
                    case["llm_error"] = str(e)
                print(f"{m['t']:6.1f}s  track {m['track_id']}  ->  {case.get('llm', case.get('llm_error'))}")

            f.write(json.dumps(case) + "\n")

    print(f"\ndone, saved to {out_file}")


if __name__ == "__main__":
    main()
