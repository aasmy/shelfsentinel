# evaluation from the proposal: recall on staged thefts, false alarm rate on normal visits and
# llm calls per visit, for the agent and for a one-call baseline. one labelled video = one visit.
#
# labels csv (see configs/labels_example.csv):
#   video,scenario,actor,theft,zones,notes
#   data/visit_001.mp4,concealment,A,yes,configs/zones.yaml,pocket left side
#
# usage:
#   python scripts/evaluate.py --labels data/labels.csv --run              # run pipeline + agent where needed, then score
#   python scripts/evaluate.py --labels data/labels.csv --run --baseline   # also the one-call baseline (1 llm call per video)
#   python scripts/evaluate.py --labels data/labels.csv                    # just score what was already run
#   python scripts/evaluate.py --labels data/labels.csv --run --no-llm     # dry run, no api calls
#
# results go to runs/eval/ (per_video.csv, summary.json). agent/baseline outputs are cached per video
# so running it again doesnt spend quota, use --rerun to redo them

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsentinel.evaluation.metrics import load_labels, score, score_by, video_verdict

METRICS = [
    ("recall_l2", "thefts caught as L2"),
    ("recall_flagged", "thefts flagged (L1 or L2)"),
    ("false_alarm_rate", "normal visits flagged (L1 or L2)"),
    ("false_alarm_rate_l2", "normal visits flagged as L2"),
    ("inconclusive_rate", "inconclusive"),
    ("llm_calls_per_visit", "llm calls per visit"),
]


def run(cmd):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def run_agent_for(row, args, out_dir):
    video = row["video"]
    stem = Path(video).stem
    events = Path(args.runs) / stem / "cases.jsonl"
    cases = out_dir / stem / "agent_cases.jsonl"
    if not events.exists() or args.rerun:
        zones = row.get("zones") or "configs/zones.yaml"
        if not run([sys.executable, "scripts/run_pipeline.py", "--video", video, "--zones", zones,
                    "--out", args.runs, "--no-llm"]):
            return None
    if not cases.exists() or args.rerun:
        cmd = [sys.executable, "scripts/run_agent.py", "--video", video, "--events", str(events),
               "--out", str(cases), "--db", str(out_dir / "eval_cases.db"),
               "--max-calls", str(args.max_calls), "--delay", str(args.delay)]
        if args.no_llm:
            cmd.append("--no-llm")
        if not run(cmd):
            return None
    return cases


def baseline_for(row, args, out_dir, llm):
    path = out_dir / Path(row["video"]).stem / "baseline.json"
    if path.exists() and not args.rerun:
        return json.loads(path.read_text())
    from shelfsentinel.evaluation.baseline import run_baseline
    from shelfsentinel.privacy.blur import blur_faces

    result = run_baseline(ROOT / row["video"], llm, n=args.baseline_frames, blur=blur_faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def print_table(title, results):
    names = list(results)
    print(f"\n{title}")
    print(f"  {'':34}" + "".join(f"{n:>14}" for n in names))
    for key, label in METRICS:
        cells = "".join(f"{'-' if results[n][key] is None else results[n][key]:>14}" for n in names)
        print(f"  {label:34}{cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--runs", default="runs", help="where run_pipeline saves its events")
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--run", action="store_true", help="run pipeline + agent for videos that werent run yet")
    parser.add_argument("--rerun", action="store_true", help="redo everything, even if cached")
    parser.add_argument("--baseline", action="store_true", help="also run the one-call baseline")
    parser.add_argument("--baseline-frames", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=3, help="max agent llm calls per video")
    parser.add_argument("--delay", type=float, default=13)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if not labels:
        sys.exit(f"no labelled videos in {args.labels}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = None
    if args.baseline and not args.no_llm:
        from shelfsentinel.llm.gemini import Gemini

        llm = Gemini()

    agent_rows, baseline_rows = [], []
    for row in labels:
        stem = Path(row["video"]).stem
        print(f"\n{stem}  ({row.get('scenario', '?')}, actor {row.get('actor', '?')}, theft={row['theft']})")

        cases_path = run_agent_for(row, args, out_dir) if args.run else out_dir / stem / "agent_cases.jsonl"
        if cases_path and Path(cases_path).exists():
            cases = [json.loads(line) for line in open(cases_path) if line.strip()]
            verdict, calls = video_verdict(cases)
            agent_rows.append({**row, "verdict": verdict, "llm_calls": calls})
            print(f"  agent: {verdict} ({calls} calls)")
        else:
            print("  agent: no results yet (use --run)")

        if args.baseline:
            cached = (out_dir / stem / "baseline.json").exists() and not args.rerun
            if llm is None and not cached:
                print("  baseline: skipped (--no-llm)")
            else:
                if not cached and baseline_rows:
                    import time

                    time.sleep(args.delay)  # free tier per minute limit
                b = baseline_for(row, args, out_dir, llm)
                baseline_rows.append({**row, "verdict": b["verdict"], "llm_calls": b["llm_calls"]})
                print(f"  baseline: {b['verdict']} ({b['reason']})")

    if not agent_rows and not baseline_rows:
        sys.exit("nothing to score yet")

    results = {}
    if agent_rows:
        results["agent"] = score(agent_rows)
    if baseline_rows:
        results["baseline"] = score(baseline_rows)
    print_table(f"overall ({len(labels)} labelled visits)", results)

    by_actor = {"agent": score_by(agent_rows, "actor")} if agent_rows else {}
    by_scenario = {"agent": score_by(agent_rows, "scenario")} if agent_rows else {}
    if agent_rows:
        print_table("agent by actor (bias audit)", by_actor["agent"])
        print_table("agent by scenario", by_scenario["agent"])

    # per video table for the report
    baseline_by_video = {r["video"]: r for r in baseline_rows}
    with open(out_dir / "per_video.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "scenario", "actor", "theft", "agent_verdict", "agent_llm_calls",
                    "baseline_verdict", "baseline_llm_calls"])
        for r in agent_rows or baseline_rows:
            b = baseline_by_video.get(r["video"], {})
            a = r if agent_rows else {}
            w.writerow([r["video"], r.get("scenario", ""), r.get("actor", ""), "yes" if r["theft"] else "no",
                        a.get("verdict", ""), a.get("llm_calls", ""), b.get("verdict", ""), b.get("llm_calls", "")])

    summary = {"overall": results, "by_actor": by_actor, "by_scenario": by_scenario}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir / 'per_video.csv'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
