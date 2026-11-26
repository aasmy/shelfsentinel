# run the case agent on a video and save the cases to runs/<video>/agent_cases.jsonl
#
# step 1 (once per video, finds the events, no llm):
#   python scripts/run_pipeline.py --video data/test.mp4 --no-llm
# step 2 (agent, uses the events from step 1):
#   python scripts/run_agent.py --video data/test.mp4 --max-calls 5
# python scripts/run_agent.py --video data/public/clips/test_vid4.mp4 --max-calls 20
#   python scripts/run_agent.py --video data/test.mp4 --no-llm      (timeline + counter check only)

import argparse
import json
import sys
import time
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.llm.provider_manager import ProviderManager
from shelfsentinel.llm.provider_status import ProviderStatus
from shelfsentinel.agent.graph import build_graph, run_case


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
    parser.add_argument("--max-calls", type=int, default=5, help="max llm calls for the whole video")
    parser.add_argument("--per-case", type=int, default=3, help="max llm calls per visit")
    parser.add_argument("--delay", type=float, default=13)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    events_path = Path(args.events or Path("runs") / Path(args.video).stem / "cases.jsonl")
    if not events_path.exists():
        sys.exit(f"{events_path} not found, run: python scripts/run_pipeline.py --video {args.video} --no-llm")
    moments = [json.loads(line) for line in open(events_path) if line.strip()]
    moments.sort(key=lambda m: m["t"])

    llm, blur = None, None
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
        from shelfsentinel.privacy.blur import blur_faces

        # llm, blur = PacedLLM(Gemini(), args.delay), blur_faces
        llm, blur = provider_manager, blur_faces

    # one visit = one track that reached into the shelf
    tracks = sorted({m["track_id"] for m in moments if m["event"] == "hand_reach_shelf" and m.get("llm_call")})
    print(f"{len(moments)} events, {len(tracks)} visits with a shelf reach\n")

    out_path = Path(args.out or Path("runs") / Path(args.video).stem / "agent_cases.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(out_path, "w")

    graph = build_graph()
    used = 0
    for case_id, tid in enumerate(tracks, 1):
        budget = min(args.per_case, args.max_calls - used)
        case = run_case(str(args.video), tid, moments, llm=llm if budget > 0 else None,
                        max_calls=max(budget, 0), blur=blur, graph=graph)
        used += case["llm_calls"]
        out.write(json.dumps(case) + "\n")
        print(f"case {case_id}  track {tid:<3} {case['severity']:<12} calls={case['llm_calls']}  {case['summary']}")
        for d in case["dropped"]:
            print(f"      dropped: {d.get('claim')} ({d['reason']})")

    out.close()
    print(f"\n{used} llm calls, cases saved to {out_path}")


if __name__ == "__main__":
    main()
