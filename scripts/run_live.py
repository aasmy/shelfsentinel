# live mode: webcam -> detection + events on screen -> when the visit ends the agent runs on it
# -> case shows up on the dashboard (run the dashboard in another terminal)
#
# usage:
#   python scripts/run_live.py                       # webcam 0, zones from configs/zones.yaml
#   python scripts/run_live.py --camera 1            # other camera
#   python scripts/run_live.py --no-llm              # no gemini calls, just detection + timeline
#   python scripts/run_live.py --source data/x.mp4   # play a video file as if it was the camera (for testing)
#
# key in the window: q = stop (the last visit still gets checked)
#
# how it works: frames are recorded into a "segment" while people are in view. when everyone has
# left (person_exited, or nobody seen for a few seconds) the segment is closed and run_agent.py runs
# on it in the background, so the camera keeps going while the agent checks the visit.

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsentinel.perception.detector import PoseDetector
from shelfsentinel.perception.events import ShelfEventDetector
from shelfsentinel.perception.zones import load_zones

ZONE_COLORS = {"shelf": (0, 200, 255), "counter": (0, 255, 0), "exit": (0, 0, 255)}
EVENT_COLORS = {"hand_to_body": (0, 0, 255), "hand_reach_shelf": (0, 200, 255), "at_counter": (0, 255, 0),
                "person_exited": (255, 255, 255)}


class Segment:
    """one visit: the frames recorded while people were around, plus their events"""

    def __init__(self, out_dir, fps, size):
        self.name = datetime.now().strftime("live_%Y%m%d_%H%M%S")
        self.dir = out_dir / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video = self.dir / f"{self.name}.mp4"
        self.writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.fps = fps
        self.frames = 0
        self.events = []
        self.detector = None  # set by the caller, a fresh event detector per visit

    @property
    def t(self):
        return self.frames / self.fps

    def close(self):
        self.writer.release()
        with open(self.dir / "cases.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")


def draw(frame, result, zones, recent, segment, status):
    img = result.plot(boxes=True, labels=True, conf=False) if result is not None else frame.copy()
    for name, polygon in zones.items():
        pts = np.array([[int(x), int(y)] for x, y in polygon], np.int32)
        color = ZONE_COLORS.get(name, (200, 200, 200))
        cv2.polylines(img, [pts], True, color, 2)
        cv2.putText(img, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    rec = f"REC {segment.t:5.1f}s" if segment else "waiting for someone"
    cv2.putText(img, f"{rec}   {status}   q=stop", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255) if segment else (200, 200, 200), 2)
    for i, (text, color) in enumerate(recent[-6:]):
        y = 64 + 30 * i
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (4, y - h - 8), (16 + w, y + 8), (0, 0, 0), -1)  # dark box so its readable
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def check_visit(segment, args):
    """run the agent on a finished visit in the background, it saves the case to the dashboard db"""
    has_reach = any(e["event"] == "hand_reach_shelf" for e in segment.events)
    if not has_reach:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="video file to use instead of the camera (for testing)")
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--out", default="runs/live")
    parser.add_argument("--db", default="runs/cases.db")
    parser.add_argument("--max-calls", type=int, default=3, help="max llm calls per visit")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--idle", type=float, default=3.0, help="seconds with nobody in view that end a visit")
    parser.add_argument("--fps", type=float, default=0, help="recording fps (default: measured)")
    parser.add_argument("--no-show", action="store_true", help="dont open a window (for testing)")
    args = parser.parse_args()

    zones = load_zones(args.zones) if Path(args.zones).exists() else {}
    if "shelf" not in zones:
        sys.exit(f"no shelf zone in {args.zones}. draw it first, e.g. record a few seconds and run "
                 "python scripts/draw_zones.py <video>")

    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        sys.exit("could not open the camera" if not args.source else f"could not open {args.source}")
    ok, frame = cap.read()
    if not ok:
        sys.exit("camera gave no frames")
    size = (frame.shape[1], frame.shape[0])

    detector = PoseDetector()
    print("warming up...")
    # recording fps = how fast we actually process frames (a file source keeps its own fps)
    fps = args.fps or (cap.get(cv2.CAP_PROP_FPS) if args.source else 0)
    if not fps:
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
    segment, last_seen, recent, running = None, None, [], []
    status = ""

    def finish(seg):
        seg.close()
        proc = check_visit(seg, args)
        if proc:
            running.append((seg, proc))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.track_frame(frame)

        if segment is None:
            t_probe = 0.0
            people = detector.get_people(result, t_probe)
            if people:  # someone walked in, start recording a visit
                segment = Segment(out_dir, fps, size)
                segment.detector = ShelfEventDetector(zones["shelf"], counter_polygon=zones.get("counter"),
                                                      exit_polygon=zones.get("exit"))
                recent = [(f"visit started ({segment.name})", (255, 255, 255))]
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
                    recent.append((f"{e['t']:5.1f}s  #{e['track_id']} {e['event']}",
                                   EVENT_COLORS.get(event["event"], (200, 200, 200))))
                    print(f"[{segment.name}] {e['t']:6.1f}s track {e['track_id']} {event['event']}")
            if people:
                last_seen = t
            visit_over = (last_seen is not None and t - last_seen >= args.idle) or (
                not people and any(e["event"] == "person_exited" for e in segment.events[-3:]))
            if visit_over:
                finish(segment)
                segment, last_seen = None, None

        # report finished checks
        for seg, proc in list(running):
            if proc.poll() is not None:
                running.remove((seg, proc))
                cases = seg.dir / "agent_cases.jsonl"
                lines = [json.loads(line) for line in open(cases)] if cases.exists() else []
                for c in lines:
                    msg = f"case: track {c['track_id']} {c['severity']} - {c['summary'][:60]}"
                    print(f"[{seg.name}] {msg}")
                    recent.append((msg, (0, 0, 255) if c["severity"] == "L2" else (0, 200, 255)))
                if not lines:
                    print(f"[{seg.name}] agent finished, see {seg.dir / 'agent.log'}")
        status = f"checking {len(running)} visit(s)" if running else ""

        if not args.no_show:
            cv2.imshow("ShelfSentinel live", draw(frame, result, zones, recent, segment, status))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if segment is not None and segment.frames:
        finish(segment)
    if not args.no_show:
        cv2.destroyAllWindows()

    if running:
        print("waiting for the agent to finish the last visit(s)...")
    for seg, proc in running:
        proc.wait()
        cases = seg.dir / "agent_cases.jsonl"
        for line in (open(cases) if cases.exists() else []):
            c = json.loads(line)
            print(f"[{seg.name}] case: track {c['track_id']} {c['severity']} - {c['summary']}")
    print(f"done. cases are in {args.db}, open the dashboard to review them")


if __name__ == "__main__":
    main()
