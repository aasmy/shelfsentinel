# live mode: webcam -> detection + events on screen -> when the visit ends the agent runs on it
# -> case shows up on screen and on the dashboard (run the dashboard in another terminal)
#
# usage:
#   python scripts/run_live.py                         # webcam 0, finds the zones by itself if theres no zones file
#   python scripts/run_live.py --auto-zones gemini     # find the zones with gemini (1 call) even if a file exists
#   python scripts/run_live.py --auto-zones yolo       # find the zones with yolo-world (free, offline)
#   python scripts/run_live.py --camera 1 --no-llm     # other camera, no gemini calls for the cases
#   python scripts/run_live.py --source data/x.mp4 --no-show   # a video file instead of the camera (testing)
#
# keys in the window: q = stop (the last visit still gets checked), z = find the zones again
#   on the zones screen: enter = use them, g = try gemini, y = try yolo-world, m = draw by hand
#
# a visit starts when someone walks in and ends when nobody has been in view for a few seconds
# (or after person_exited). then run_agent.py checks it in the background so the camera keeps going.

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsentinel.live_ui import compose
from shelfsentinel.perception.detector import PoseDetector
from shelfsentinel.perception.events import ShelfEventDetector
from shelfsentinel.perception.zones import load_zones, save_zones

WINDOW = "ShelfSentinel live"
RANK = {"L2": 3, "L1": 2, "inconclusive": 1}


class Segment:
    """one visit: the frames recorded while people were around, plus their events"""

    def __init__(self, out_dir, fps, size, zones):
        self.name = datetime.now().strftime("live_%Y%m%d_%H%M%S")
        self.dir = out_dir / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video = self.dir / f"{self.name}.mp4"
        self.writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.fps = fps
        self.frames = 0
        self.events = []
        self.detector = ShelfEventDetector(zones["shelf"], counter_polygon=zones.get("counter"),
                                           exit_polygon=zones.get("exit"))
        save_zones(zones, str(self.dir / "zones.yaml"))  # keep the zones used for this visit

    @property
    def t(self):
        return self.frames / self.fps

    def close(self):
        self.writer.release()
        with open(self.dir / "cases.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")


def check_visit(segment, args):
    """run the agent on a finished visit in the background, it saves the case to the dashboard db"""
    if not any(e["event"] == "hand_reach_shelf" for e in segment.events):
        print(f"[{segment.name}] no shelf reach, nothing to check")
        return None
    cmd = [sys.executable, str(ROOT / "scripts" / "run_agent.py"), "--video", str(segment.video),
           "--events", str(segment.dir / "cases.jsonl"), "--out", str(segment.dir / "agent_cases.jsonl"),
           "--db", args.db, "--max-calls", str(args.max_calls)]
    if args.no_llm:
        cmd.append("--no-llm")
    print(f"[{segment.name}] visit over ({segment.t:.1f}s, {len(segment.events)} events), checking it...")
    log = open(segment.dir / "agent.log", "w")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def default_method(args):
    return "yolo" if args.no_llm or not os.getenv("GEMINI_API_KEY") else "gemini"


def find_zones(frame, method):
    """returns (zones, message)"""
    try:
        if method == "gemini":
            from shelfsentinel.llm.gemini import Gemini
            from shelfsentinel.perception.auto_zones import find_zones_gemini
            from shelfsentinel.privacy.blur import blur_faces

            zones, _ = find_zones_gemini(frame, Gemini(), blur=blur_faces)
        else:
            from shelfsentinel.perception.auto_zones import find_zones_yolo

            zones, _ = find_zones_yolo(frame)
    except Exception as e:
        return {}, f"{method} failed: {str(e)[:60]}"
    missing = [n for n in ("shelf", "counter", "exit") if n not in zones]
    return zones, f"found by {method}" + (f", missing {', '.join(missing)}" if missing else "")


def draw_by_hand(frame):
    """opens the normal draw_zones tool on this frame, returns the zones it saved"""
    image = ROOT / "runs" / "zones_frame.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), frame)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "draw_zones.py"), str(image)], cwd=ROOT)
    path = ROOT / "configs" / "zones.yaml"  # where draw_zones.py saves
    return load_zones(str(path)) if path.exists() else {}


def setup_zones(cap, method, args):
    """find the zones on the current camera view and let the user accept or change them"""
    ok, frame = cap.read()
    if not ok:
        return {}, "camera gave no frame"
    print(f"finding zones with {method}...")
    zones, message = find_zones(frame, method)
    print(message)
    if args.no_show:
        return zones, message

    while True:
        ok, live = cap.read()
        if ok:
            frame = live
        status = "SETUP" if "shelf" in zones else "SETUP  no shelf"
        img = compose(frame, zones, None, status, "", [], None, message,
                      "enter use  g gemini  y yolo  m draw  q quit")
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and "shelf" in zones:
            return zones, message
        if key == ord("g"):
            zones, message = find_zones(frame, "gemini")
        elif key == ord("y"):
            zones, message = find_zones(frame, "yolo")
        elif key == ord("m"):
            drawn = draw_by_hand(frame)
            if drawn:
                zones, message = drawn, "drawn by hand"
        elif key == ord("q"):
            sys.exit("stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="video file to use instead of the camera (for testing)")
    parser.add_argument("--zones", default="configs/zones_live.yaml", help="zones file to use / save")
    parser.add_argument("--auto-zones", choices=["gemini", "yolo"],
                        help="find the zones automatically at the start (default: only if the zones file is missing)")
    parser.add_argument("--out", default="runs/live")
    parser.add_argument("--db", default="runs/cases.db")
    parser.add_argument("--max-calls", type=int, default=3, help="max llm calls per visit")
    parser.add_argument("--no-llm", action="store_true", help="no gemini calls for checking visits")
    parser.add_argument("--idle", type=float, default=3.0, help="seconds with nobody in view that end a visit")
    parser.add_argument("--fps", type=float, default=0, help="recording fps (default: measured)")
    parser.add_argument("--no-show", action="store_true", help="dont open a window (for testing)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        sys.exit("could not open the camera" if not args.source else f"could not open {args.source}")
    ok, frame = cap.read()
    if not ok:
        sys.exit("camera gave no frames")
    size = (frame.shape[1], frame.shape[0])

    if not args.no_show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1280, 720)

    # zones: from the file, or found automatically
    zones_path = ROOT / args.zones
    zones = load_zones(str(zones_path)) if zones_path.exists() and not args.auto_zones else {}
    zones_source = f"from {args.zones}" if zones else ""
    if "shelf" not in zones:
        zones, zones_source = setup_zones(cap, args.auto_zones or default_method(args), args)
        if "shelf" not in zones:
            sys.exit(f"no shelf zone found ({zones_source}). try --auto-zones gemini, or draw them by hand")
        zones_path.parent.mkdir(parents=True, exist_ok=True)
        save_zones(zones, str(zones_path))
        print(f"zones saved to {args.zones}")

    detector = PoseDetector()
    print("warming up...")
    fps = args.fps or (cap.get(cv2.CAP_PROP_FPS) if args.source else 0)
    if not fps:  # recording fps = how fast we actually process frames
        start, n = time.time(), 0
        while n < 15:
            ok, frame = cap.read()
            if not ok:
                break
            detector.track_frame(frame)
            n += 1
        fps = max(2.0, round(n / (time.time() - start), 1))
    print(f"running at {fps} fps. zones: {', '.join(zones)}. press q in the window to stop")

    out_dir = ROOT / args.out
    segment, last_seen = None, None
    events, running, latest_case = [], [], None

    def finish(seg):
        seg.close()
        proc = check_visit(seg, args)
        if proc:
            running.append((seg, proc))

    def collect(seg):
        nonlocal latest_case
        cases_file = seg.dir / "agent_cases.jsonl"
        cases = [json.loads(line) for line in open(cases_file)] if cases_file.exists() else []
        for c in cases:
            print(f"[{seg.name}] case: track {c['track_id']} {c['severity']} - {c['summary']}")
        if cases:
            latest_case = max(cases, key=lambda c: RANK.get(c["severity"], 0))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.track_frame(frame)

        if segment is None and detector.get_people(result, 0.0):
            segment = Segment(out_dir, fps, size, zones)  # someone walked in
            events = []
            print(f"[{segment.name}] someone walked in, recording")

        if segment is not None:
            t = segment.t
            people = detector.get_people(result, t)
            segment.writer.write(frame)
            segment.frames += 1
            for event in segment.detector.process_frame(people, round(t, 3)):
                e = dict(event)
                e["t"] = round(event["timestamp"], 2)
                segment.events.append(e)
                # skip the noise: leaves, and reaches that are just the wrist flickering on the shelf edge
                if event["event"] != "hand_leaves_shelf" and not (
                        event["event"] == "hand_reach_shelf" and not event["llm_call"]):
                    events.append((e["t"], e["track_id"], event["event"]))
                    print(f"[{segment.name}] {e['t']:6.1f}s track {e['track_id']} {event['event']}")
            if people:
                last_seen = t
            visit_over = (last_seen is not None and t - last_seen >= args.idle) or (
                not people and any(e["event"] == "person_exited" for e in segment.events[-3:]))
            if visit_over:
                finish(segment)
                segment, last_seen = None, None

        for seg, proc in list(running):  # finished checks
            if proc.poll() is not None:
                running.remove((seg, proc))
                collect(seg)

        if not args.no_show:
            if segment is not None:
                status, timer = "RECORDING", f"{segment.t:5.1f}s"
            elif running:
                status, timer = f"CHECKING {len(running)}", ""
            else:
                status, timer = "WAITING", ""
            img = compose(frame, zones, result, status, timer, events, latest_case, zones_source,
                          "q stop    z find zones again")
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("z"):
                if segment is not None:  # close the current visit first, zones change from here
                    finish(segment)
                    segment, last_seen = None, None
                new_zones, source = setup_zones(cap, default_method(args), args)
                if "shelf" in new_zones:
                    zones, zones_source = new_zones, source
                    save_zones(zones, str(zones_path))

    cap.release()
    if segment is not None and segment.frames:
        finish(segment)
    if not args.no_show:
        cv2.destroyAllWindows()
    if running:
        print("waiting for the agent to finish the last visit(s)...")
    for seg, proc in running:
        proc.wait()
        collect(seg)
    print(f"done. cases are in {args.db}, open the dashboard to review them")


if __name__ == "__main__":
    main()
