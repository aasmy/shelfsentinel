"""store scenarios: what the agent decides when the llm reports a sequence of actions.
the llm is scripted (no api), frames are fake, so these test the agent's reasoning only."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import shelfsentinel.agent.graph as agent_graph
from shelfsentinel.agent.graph import build_graph, follow_items, run_case

GRAPH = build_graph()


class ScriptedLLM:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def ask_with_images(self, prompt, frames):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else {"clear_view": True, "claims": []}


@pytest.fixture(autouse=True)
def fake_frames(monkeypatch):
    windows = []

    def get_keyframes(video, t0, t1, n=5):
        windows.append((t0, t1, n))
        return [(round(t0 + (t1 - t0) * i / (n - 1), 2), f"f{i}") for i in range(n)]

    monkeypatch.setattr(agent_graph, "get_keyframes", get_keyframes)
    return windows


def ev(name, t, llm_call=False):
    return {"track_id": 1, "event": name, "t": t, "llm_call": llm_call}


def said(*claims):
    """llm answer with these claims, on frames 1, 2, 3 ... in order"""
    return {"clear_view": True,
            "claims": [{"claim": c, "frame": i + 1, "what": c.replace("_", " ")} for i, c in enumerate(claims)]}


def visit(*extra, exit_t=20.0, counter_t=None):
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0), *extra]
    if counter_t is not None:
        moments.append(ev("at_counter", counter_t))
    if exit_t is not None:
        moments.append(ev("person_exited", exit_t))
    return sorted(moments, key=lambda m: m["t"])


def run(moments, answers, max_calls=3):
    llm = ScriptedLLM(answers)
    case = run_case("v.mp4", 1, moments, llm=llm, max_calls=max_calls, graph=GRAPH)
    return case, llm


# --- one reach, different things happen


def test_took_and_hid_then_left_is_l2():
    case, _ = run(visit(), [said("item_taken", "item_hidden")])
    assert case["severity"] == "L2"


def test_took_and_put_back_then_left_is_normal():
    case, _ = run(visit(), [said("item_taken", "item_put_back")])
    assert case["severity"] == "normal"
    assert "put them back" in case["summary"]


def test_hid_then_took_out_and_put_back_is_normal():
    case, _ = run(visit(), [said("item_taken", "item_hidden", "item_taken_out", "item_put_back")])
    assert case["severity"] == "normal"
    assert "taken out again" in case["summary"]
    assert case["checks"]["hidden_then_taken_out"] is True


def test_hid_then_took_out_but_left_holding_it_is_l1():
    case, _ = run(visit(), [said("item_taken", "item_hidden", "item_taken_out")])
    assert case["severity"] == "L1"
    assert "taken out" in case["summary"]


def test_took_and_left_holding_it_is_l1():
    case, _ = run(visit(), [said("item_taken")])
    assert case["severity"] == "L1"


def test_rummaging_without_taking_is_normal():
    case, _ = run(visit(), [{"clear_view": True, "claims": []}])
    assert case["severity"] == "normal"


def test_hid_but_went_to_counter_is_l1_not_l2():
    # hiding then paying is still worth a look (could pay for something else), but not L2
    moments = visit(ev("hand_to_body", 3.5), counter_t=15.0, exit_t=None)
    case, _ = run(moments, [said("item_taken", "item_hidden")])
    assert case["severity"] == "L1"


def test_hid_and_still_in_store_is_l1():
    case, _ = run(visit(exit_t=None), [said("item_taken", "item_hidden")])
    assert case["severity"] == "L1"
    assert "exit not seen" in case["summary"]


# --- the action is spread over several moments


def test_took_two_put_one_back_left_is_l1():
    case, _ = run(visit(), [said("item_taken", "item_taken", "item_put_back")])
    assert case["severity"] == "L1"


def test_took_at_one_reach_put_back_at_a_later_reach_is_normal():
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0),
               ev("hand_reach_shelf", 12.0, True), ev("hand_leaves_shelf", 13.0), ev("person_exited", 25.0)]
    case, llm = run(moments, [said("item_taken"), said("item_put_back")])
    assert len(llm.prompts) == 2
    assert case["severity"] == "normal"


def test_pocketed_later_away_from_the_shelf_is_still_seen():
    # hand to pocket 8s after leaving the shelf: outside the reach window, gets its own look
    moments = visit(ev("hand_to_body", 11.0))
    case, llm = run(moments, [said("item_taken"), said("item_hidden")])
    assert len(llm.prompts) == 2
    assert "hand to their body" in llm.prompts[1]  # the pocket moment gets its own look
    assert case["severity"] == "L2"


def test_hid_at_shelf_then_took_out_later_and_put_back():
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 3.0), ev("hand_to_body", 4.0),
               ev("hand_reach_shelf", 15.0, True), ev("hand_leaves_shelf", 16.0), ev("person_exited", 30.0)]
    case, _ = run(moments, [said("item_taken", "item_hidden"), said("item_taken_out", "item_put_back")])
    assert case["severity"] == "normal"


def test_put_back_straight_from_pocket():
    case, _ = run(visit(), [said("item_taken", "item_hidden", "item_put_back")])
    assert case["severity"] == "normal"


def test_rummaging_quick_reaches_use_one_call():
    moments = [ev("hand_reach_shelf", 2.0, True), ev("hand_leaves_shelf", 2.5),
               ev("hand_reach_shelf", 3.5, True), ev("hand_leaves_shelf", 4.0),
               ev("hand_reach_shelf", 5.0, True), ev("hand_leaves_shelf", 5.5),
               ev("hand_reach_shelf", 6.5, True), ev("hand_leaves_shelf", 7.0), ev("person_exited", 20.0)]
    case, llm = run(moments, [said("item_taken", "item_put_back")])
    assert len(llm.prompts) == 1
    assert case["severity"] == "normal"


# --- bad / unclear answers stay careful


def test_hidden_with_unclear_view_is_not_trusted():
    answer = {"clear_view": False, "claims": [{"claim": "item_hidden", "frame": 2, "what": "puts it in a bag"}]}
    case, _ = run(visit(), [answer, answer])
    assert case["severity"] == "inconclusive"


def test_same_action_seen_twice_on_the_second_look_counts_once():
    unclear = {"clear_view": False, "claims": [{"claim": "item_taken", "frame": 1, "what": "takes a box"}]}
    again = {"clear_view": True, "claims": [{"claim": "item_taken", "frame": 1, "what": "takes a box"},
                                            {"claim": "item_put_back", "frame": 5, "what": "puts it back"}]}
    case, _ = run(visit(), [unclear, again])
    assert [e["claim"] for e in case["evidence"]] == ["item_taken", "item_put_back"]
    assert case["severity"] == "normal"


def test_follow_items_counts():
    seq = [{"claim": c, "frame_time": t} for t, c in enumerate(
        ["item_taken", "item_taken", "item_hidden", "item_put_back", "item_taken_out"])]
    assert follow_items(seq) == (1, 0, True, 1)  # one in hand, nothing hidden, was hidden, one put back
