"""evaluation scoring and the one-call baseline, with a fake llm"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shelfsentinel.evaluation.baseline import run_baseline
from shelfsentinel.evaluation.metrics import load_labels, score, score_by, video_verdict


def test_video_verdict_takes_the_worst_case():
    cases = [{"severity": "normal", "llm_calls": 0}, {"severity": "L2", "llm_calls": 2},
             {"severity": "inconclusive", "llm_calls": 1}]
    assert video_verdict(cases) == ("L2", 3)
    assert video_verdict([]) == ("normal", 0)


def test_score():
    rows = [
        {"theft": True, "verdict": "L2", "llm_calls": 2},
        {"theft": True, "verdict": "L1", "llm_calls": 1},
        {"theft": True, "verdict": "inconclusive", "llm_calls": 3},
        {"theft": False, "verdict": "normal", "llm_calls": 0},
        {"theft": False, "verdict": "L1", "llm_calls": 2},
    ]
    s = score(rows)
    assert s["recall_l2"] == round(1 / 3, 3)
    assert s["recall_flagged"] == round(2 / 3, 3)
    assert s["false_alarm_rate"] == 0.5
    assert s["false_alarm_rate_l2"] == 0.0
    assert s["llm_calls_per_visit"] == 1.6


def test_score_by_actor():
    rows = [{"theft": False, "verdict": "L1", "llm_calls": 1, "actor": "A"},
            {"theft": False, "verdict": "normal", "llm_calls": 0, "actor": "B"}]
    by = score_by(rows, "actor")
    assert by["A"]["false_alarm_rate"] == 1.0 and by["B"]["false_alarm_rate"] == 0.0


def test_labels_example_loads():
    rows = load_labels(ROOT / "configs" / "labels_example.csv")
    assert len(rows) == 4 and [r["theft"] for r in rows] == [False, False, False, True]


def make_video(path, seconds=4, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(seconds * fps):
        w.write(np.full((32, 32, 3), i % 255, np.uint8))
    w.release()
    return path


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls, self.frames = answer, 0, 0

    def ask_with_images(self, prompt, frames):
        self.calls += 1
        self.frames = len(frames)
        return self.answer


def test_baseline_one_call(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    llm = FakeLLM([{"theft": "true", "clear_view": True, "reason": "puts box in pocket"}])
    result = run_baseline(video, llm, n=5)
    assert llm.calls == 1 and llm.frames == 5
    assert result["verdict"] == "L2" and result["llm_calls"] == 1
    assert run_baseline(video, FakeLLM({"theft": False, "clear_view": False}))["verdict"] == "inconclusive"
    assert run_baseline(video, FakeLLM({"theft": False, "clear_view": True}))["verdict"] == "normal"


def test_evaluate_script_scores_cached_results(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("video,scenario,actor,theft\nv1.mp4,concealment,A,yes\nv2.mp4,normal,B,no\n")
    out = tmp_path / "eval"
    for stem, severity in (("v1", "L2"), ("v2", "normal")):
        (out / stem).mkdir(parents=True)
        (out / stem / "agent_cases.jsonl").write_text(json.dumps({"severity": severity, "llm_calls": 1}) + "\n")
        (out / stem / "baseline.json").write_text(json.dumps({"verdict": "normal", "llm_calls": 1, "reason": "x"}))

    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--labels", str(labels),
                           "--out", str(out), "--baseline", "--no-llm"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["overall"]["agent"]["recall_l2"] == 1.0
    assert summary["overall"]["baseline"]["recall_l2"] == 0.0
    assert (out / "per_video.csv").read_text().count("\n") == 3
