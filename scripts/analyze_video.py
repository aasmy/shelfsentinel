# analyze one video end to end: zones -> events -> case agent -> database (what the dashboard upload runs)
#
#   python scripts/analyze_video.py --video data/uploads/test.mp4 --auto-zones yolo --zones-only
#       (just find the zones and save a preview picture)
#   python scripts/analyze_video.py --video data/uploads/test.mp4 --zones runs/test/zones.yaml
#   python scripts/analyze_video.py --video data/uploads/test.mp4 --auto-zones gemini --no-llm
#
# everything for the video goes to runs/<video name>/ (zones.yaml, zones.png, events, cases, log)

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsentinel.perception.zones import load_zones, save_zones
from shelfsentinel.storage.db import DEFAULT_DB


def middle_frame(video):
    """a frame from a bit into the video, the first one is often black or blurry"""
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(30, max(count // 10, 0)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def preview(frame, zones, path):
    from shelfsentinel.live_ui import draw_zones

    cv2.imwrite(str(path), draw_zones(frame.copy(), zones, alpha=0.25))


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        sys.exit(f"step failed ({result.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", help="zones file to use (otherwise found automatically)")
    parser.add_argument("--auto-zones", choices=["gemini", "yolo"], default="yolo")
    parser.add_argument("--zones-only", action="store_true", help="only find the zones and save a preview")
    parser.add_argument("--no-llm", action="store_true", help="dont ask gemini about the reaches")
    parser.add_argument("--max-calls", type=int, default=5)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()

    video = Path(args.video)
    out = ROOT / "runs" / video.stem
    out.mkdir(parents=True, exist_ok=True)
    zones_path = out / "zones.yaml"

    frame = middle_frame(video)
    if frame is None:
        sys.exit(f"could not read {video}")

    print("1) zones", flush=True)
    if args.zones:
        zones, message = load_zones(args.zones), f"from {Path(args.zones).name}"
    else:
        from shelfsentinel.perception.auto_zones import find_zones_with

        zones, message = find_zones_with(frame, args.auto_zones)
    if "shelf" not in zones:
        sys.exit(f"no shelf zone ({message})")
    h, w = frame.shape[:2]
    from shelfsentinel.zone_editor import clamp_zones

    zones = clamp_zones(zones, w, h)
    save_zones(zones, str(zones_path))
    preview(frame, zones, out / "zones.png")
    print(f"zones {message}: {', '.join(zones)}", flush=True)
    if args.zones_only:
        return

    print("2) finding events (pose + tracking, no llm)", flush=True)
    run([sys.executable, "scripts/run_pipeline.py", "--video", video, "--zones", zones_path, "--no-llm"])

    print("3) case agent", flush=True)
    agent = [sys.executable, "scripts/run_agent.py", "--video", video, "--db", args.db,
             "--max-calls", args.max_calls]
    if args.no_llm:
        agent.append("--no-llm")
    run(agent)
    print("done", flush=True)


if __name__ == "__main__":
    main()
