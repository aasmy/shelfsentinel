# run the case agent on a video and save the cases to runs/<video>/agent_cases.jsonl
#
# step 1 (once per video, finds the events, no llm):
#   python scripts/run_pipeline.py --video data/test.mp4 --no-llm
# step 2 (agent, uses the events from step 1):
#   python scripts/run_agent.py --video data/test.mp4 --max-calls 5
#   python scripts/run_agent.py --video data/test.mp4 --no-llm      (timeline + counter check only)

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.agent.graph import build_graph, run_case
from shelfsentinel.storage.db import (
    DEFAULT_DB,
    connect,
    save_case,
    update_case_video_clips,
)
from shelfsentinel.video.clips import create_case_clips


class PacedLLM:
    """waits between calls so we stay under the free tier per minute limit"""

    def __init__(self, llm, delay):
        self.llm, self.delay, self.last = llm, delay, None

    def ask_with_images(self, prompt, frames):
        if self.last is not None:
            wait = self.delay - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
        self.last = time.time()
        return self.llm.ask_with_images(prompt, frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", help="events file from run_pipeline (default runs/<video>/cases.jsonl)")
    parser.add_argument("--out", help="where to save the cases (default runs/<video>/agent_cases.jsonl)")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--max-calls", type=int, default=5, help="max llm calls for the whole video")
    parser.add_argument("--per-case", type=int, default=3, help="max llm calls per visit")
    parser.add_argument("--delay", type=float, default=13)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--full-visit", action="store_true", help="also save the whole visit as a clip (live mode)")
    args = parser.parse_args()

    events_path = Path(args.events or Path("runs") / Path(args.video).stem / "cases.jsonl")
    if not events_path.exists():
        sys.exit(f"{events_path} not found, run: python scripts/run_pipeline.py --video {args.video} --no-llm")
    moments = [json.loads(line) for line in open(events_path) if line.strip()]
    moments.sort(key=lambda m: m["t"])

    llm, blur = None, None
    if not args.no_llm:
        from shelfsentinel.llm.gemini import Gemini
        from shelfsentinel.privacy.blur import blur_faces

        llm, blur = PacedLLM(Gemini(), args.delay), blur_faces

    # one visit = one track that reached into the shelf
    tracks = sorted({m["track_id"] for m in moments if m["event"] == "hand_reach_shelf" and m.get("llm_call")})
    # people with a hand_to_body event get checked first, so the llm budget goes to them
    suspicious = {m["track_id"] for m in moments if m["event"] == "hand_to_body"}
    tracks.sort(key=lambda tid: tid not in suspicious)
    print(f"{len(moments)} events, {len(tracks)} visits with a shelf reach\n")

    out_path = Path(args.out or Path("runs") / Path(args.video).stem / "agent_cases.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    graph = build_graph()
    used = 0
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for case_number, tid in enumerate(tracks, 1):
                budget = min(args.per_case, args.max_calls - used)
                case = run_case(
                    str(args.video),
                    tid,
                    moments,
                    llm=llm if budget > 0 else None,
                    max_calls=max(budget, 0),
                    blur=blur,
                    graph=graph,
                )
                used += case["llm_calls"]

                # Persist first so the stable database ID owns the evidence folder.
                db_id = save_case(conn, case)
                clips = []
                if case.get("checks"):
                    clips = create_case_clips(
                        args.video,
                        db_id,
                        tid,
                        moments,
                        case["evidence"],
                        **({"full_visit": True} if args.full_visit else {}),
                    )
                update_case_video_clips(conn, db_id, clips)
                case["video_clips"] = clips

                out.write(json.dumps(case) + "\n")
                print(
                    f"case {case_number} (db {db_id})  track {tid:<3} "
                    f"{case['severity']:<12} calls={case['llm_calls']}  "
                    f"{case['summary']}"
                )
                for dropped in case["dropped"]:
                    print(
                        f"      dropped: {dropped.get('claim')} "
                        f"({dropped['reason']})"
                    )
    finally:
        conn.close()
    print(f"\n{used} llm calls, cases saved to {out_path} and {args.db}")


if __name__ == "__main__":
    main()
