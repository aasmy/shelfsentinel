"""agent tests with a fake llm and a tiny generated video, no api calls"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.agent.graph import build_graph, run_case


class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def ask_with_images(self, prompt, frames):
        self.calls += 1
        return self.answers.pop(0) if self.answers else {"clear_view": False, "claims": []}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "v.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(300):
        w.write(np.full((32, 32, 3), i % 255, np.uint8))
    w.release()
    return path


def ev(event, t, llm_call=False, tid=1):
    return {"track_id": tid, "event": event, "t": t, "llm_call": llm_call}


HIDDEN = {"clear_view": True, "claims": [{"claim": "item_taken", "frame": 1, "what": "takes a box"},
                                         {"claim": "item_hidden", "frame": 4, "what": "puts the box in a pocket"}]}
GRAPH = build_graph()


def test_hidden_and_no_counter_is_l2(video):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0), ev("person_exited", 12.0)]
    llm = FakeLLM([HIDDEN])
    case = run_case(video, 1, moments, llm=llm, graph=GRAPH)
    assert case["severity"] == "L2"
    assert llm.calls == 1
    assert all("frame_time" in e for e in case["evidence"])


def test_hidden_but_paid_is_not_l2(video):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0),
               ev("at_counter", 8.0), ev("person_exited", 12.0)]
    case = run_case(video, 1, moments, llm=FakeLLM([HIDDEN]), graph=GRAPH)
    assert case["severity"] == "normal"  # normal path, agent doesnt even look
    assert case["llm_calls"] == 0


def test_unclear_view_looks_again_then_inconclusive(video):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0), ev("person_exited", 12.0)]
    llm = FakeLLM([{"clear_view": False, "claims": []}, {"clear_view": False, "claims": []}])
    case = run_case(video, 1, moments, llm=llm, graph=GRAPH)
    assert llm.calls == 2  # looped back once for more frames
    assert case["severity"] == "inconclusive"


def test_hedged_claim_is_dropped(video):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0), ev("person_exited", 12.0)]
    llm = FakeLLM([{"clear_view": True, "claims": [
        {"claim": "item_hidden", "frame": 3, "what": "appears to put it in a bag"}]}])
    case = run_case(video, 1, moments, llm=llm, graph=GRAPH)
    assert case["severity"] == "normal"
    assert case["dropped"][0]["reason"] == "hedged, not seen"


def test_respects_max_calls(video):
    moments = [ev("hand_reach_shelf", t, True) for t in (1.0, 5.0, 9.0, 13.0)] + [ev("person_exited", 20.0)]
    llm = FakeLLM([])
    run_case(video, 1, moments, llm=llm, max_calls=2, graph=GRAPH)
    assert llm.calls == 2


def test_no_llm_gives_inconclusive(video):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("person_exited", 12.0)]
    case = run_case(video, 1, moments, llm=None, graph=GRAPH)
    assert case["severity"] == "inconclusive" and case["llm_calls"] == 0
