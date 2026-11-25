# full rough pipeline: video -> detector -> hand on shelf moments -> frames -> gemini -> results
# usage: python scripts/run_pipeline.py --video data/test.mp4
#        python scripts/run_pipeline.py --video data/test.mp4 --no-llm   (just find the moments, no api calls)
# python scripts/run_pipeline.py --video data/public/clips/1_1_crop.mp4
# python scripts/run_pipeline.py --video data/public/clips/test_vid4.mp4
# saves runs/<video>/cases.jsonl

import argparse
import json
import sys
import time
from pathlib import Path

import cv2



sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.frames import get_keyframes, reach_window
from shelfsentinel.llm.provider_manager import ProviderManager
from shelfsentinel.llm.provider_status import ProviderStatus
from shelfsentinel.perception.detector import PoseDetector
from shelfsentinel.perception.events import ShelfEventDetector
from shelfsentinel.perception.zones import load_zones
from shelfsentinel.privacy.blur import blur_faces

PROMPT = (
    "These are {n} frames from a store camera, in order, taken at {times} seconds. "
    "They start when a person reaches into the shelf and continue for a few seconds after their hand leaves it. "
    "Focus on the person whose hand is at the shelf. "
    "Describe only what their hands do with items (picking up, putting back, putting in pocket or bag). "
    "Do not describe or guess who the person is. "
    'Answer as JSON: {{"hands": "...", "item_taken": true/false, "item_hidden": true/false, "clear_view": true/false}}'
)


def find_shelf_moments(
    detector,
    video_path,
    shelf,
    fps,
    counter_polygon=None,
    exit_polygon=None,
    min_frames=3,
    cooldown=3.0,
):
    event_detector = ShelfEventDetector(
        shelf,
        counter_polygon=counter_polygon,
        exit_polygon=exit_polygon,
        min_frames=min_frames,
        suppression_seconds=cooldown,
    )
    moments = []
    for frame, result in enumerate(detector.track(video_path)):
        t = frame / fps
        people = detector.get_people(result, t)
        for event in event_detector.process_frame(people, round(t, 3)):
            moment = dict(event)
            moment["t"] = round(event["timestamp"], 2)
            moments.append(moment)
            print(
                f"{moment['t']:6.1f}s  track {moment['track_id']} "
                f"{moment['event']} llm_call={moment['llm_call']}"
            )

        if frame % 300 == 0:
            print(f"... {t:.0f}s")

    return moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--max-calls", type=int, default=100, help="max gemini calls per run, to save quota")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--delay", type=float, default=10, help="seconds between gemini calls (free tier is 5 per minute)")
    args = parser.parse_args()

    zones = load_zones(args.zones)
    if "shelf" not in zones:
        sys.exit(f"no shelf zone in {args.zones}, run draw_zones.py first")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    print("finding events:")
    moments = find_shelf_moments(
        PoseDetector(),
        args.video,
        zones["shelf"],
        fps,
        counter_polygon=zones.get("counter"),
        exit_polygon=zones.get("exit"),
    )
    print(f"found {len(moments)} moments\n")

    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cases.jsonl"

    provider_manager = None
    if not args.no_llm:
        from shelfsentinel.llm.gemini import Gemini
        from shelfsentinel.llm.groq_api import GroqApi
        gemini = Gemini()
        groq_api = GroqApi()
        providers = [

            ProviderStatus("gemini", gemini),
            ProviderStatus("groq", groq_api),
        ]
        provider_manager = ProviderManager(providers)


    print("2) asking gemini" if gemini else "2) skipping gemini (--no-llm)")

    with open(out_file, "w") as f:
        for i, m in enumerate(moments):
            case = dict(m)
            t0, t1 = reach_window(moments, i)
            frames = get_keyframes(args.video, t0, t1, n=5)
            case["frame_times"] = [ft for ft, _ in frames]

            if provider_manager is not None and i < args.max_calls:
                # if i > 0:
                #
                case["llm"] = ""
                prompt = PROMPT.format(n=len(frames), times=case["frame_times"])
                case["llm"] = provider_manager.ask_with_images(prompt,
                                                     [fr for _, fr in frames])
                print(f"{m['t']:6.1f}s  track {m['track_id']}  ->  {case.get('llm')}")


            f.write(json.dumps(case) + "\n")

    print(f"\ndone, saved to {out_file}")


if __name__ == "__main__":
    main()
