# case agent: one visit (track) in, one case out
#
# build_timeline -> inspect_reaches -> (view not clear? look again with more frames) -> check_counter -> decide
#
# only visits that dont follow reach -> counter -> exit get sent to the llm, normal visits cost 0 calls.
# what the llm sees is followed in time order (take, put back, hide, take out again), so someone who
# picks something up and puts it back, or hides it and takes it out again, doesnt get flagged

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from shelfsentinel.frames import get_keyframes, reach_window
from shelfsentinel.guardrail.verify import clean_text, verify

EXIT_EVENTS = {"person_exited", "exit"}

WHEN = {
    "reach": "They start when a person reaches into the shelf and continue after their hand leaves it. ",
    "body": "They are around a moment the person moved a hand to their body (pocket, bag, waist or jacket). ",
}

PROMPT = (
    "These are {n} frames from a store camera, in order, numbered 1 to {n}, taken at {times} seconds. "
    "{when}"
    "Faces are blurred on purpose. Only describe what the hands do with items. "
    "Do not describe the person. Only report what you can actually see, not what might have happened. "
    "Claims: item_taken = an item is lifted off the shelf and stays in the hand. "
    "item_put_back = an item is put back on the shelf. "
    "item_hidden = an item goes into a pocket, bag, clothing or somewhere out of sight. "
    "item_taken_out = an item comes back out of a pocket, bag or clothing. "
    "Touching, moving or looking through items on the shelf without lifting one off is not item_taken. "
    "Report every action in the order it happens, one claim each, also when the same item is taken and "
    "then put back. "
    "Answer as JSON: "
    '{{"clear_view": true/false (can you see the hands well enough to tell?), '
    '"claims": [{{"claim": "item_taken" | "item_put_back" | "item_hidden" | "item_taken_out", '
    '"frame": number of the frame that shows it, "what": "short description"}}]}}. '
    "Use an empty claims list if nothing happened."
)

MAX_WINDOW = 12.0  # seconds, longest stretch shown in one llm call (rummaging = many quick reaches)


class CaseState(TypedDict, total=False):
    video: str
    track_id: int
    moments: list  # all events from the video, in time order
    llm: object  # anything with ask_with_images(prompt, frames) -> dict, or None
    max_calls: int
    blur: object  # function frame -> frame

    timeline: list
    windows: list  # stretches of video to show the llm, most suspicious first
    flagged: int  # how many of them have a hand_to_body (hand to pocket / bag)
    unclear: list  # windows the llm couldnt see clearly, checked again on the second look
    deviation: bool
    attempt: int
    llm_calls: int
    clear_view: bool
    evidence: list
    dropped: list
    counter_visit: object  # True / False / None (no counter events in this video)
    exited: bool
    severity: str
    summary: str
    checks: dict


def build_timeline(state):
    tid = state["track_id"]
    timeline = [m for m in state["moments"] if m["track_id"] == tid]
    reaches = [i for i, m in enumerate(state["moments"]) if m["track_id"] == tid
               and m["event"] == "hand_reach_shelf" and m.get("llm_call")]

    # normal path: reached the shelf, then counter, then left. no need to look closer
    counter_t = [m["t"] for m in timeline if m["event"] == "at_counter"]
    exit_t = [m["t"] for m in timeline if m["event"] in EXIT_EVENTS]
    normal = bool(reaches) and counter_t and exit_t and max(counter_t) > state["moments"][reaches[-1]]["t"] \
        and max(exit_t) > max(counter_t)

    windows = plan_windows(state["moments"], reaches, tid)
    return {"timeline": timeline, "windows": windows, "flagged": sum(w["flagged"] for w in windows),
            "deviation": bool(reaches) and not normal,
            "attempt": 0, "llm_calls": 0, "evidence": [], "dropped": [], "clear_view": False}


def plan_windows(moments, reaches, tid):
    """which parts of the visit to show the llm, in the order to check them (calls are limited):

    1. reaches followed by a hand to the body (most likely hiding)
    2. hand to body moments away from the shelf (hid it later, or took it back out)
    3. the other reaches
    reaches whose windows overlap (rummaging, quick grabs) are shown together in one call.
    """
    wins = []
    for i in sorted(reaches, key=lambda i: moments[i]["t"]):
        t0, t1 = reach_window(moments, i)
        flagged = followed_by_hand_to_body(moments, i)
        last = wins[-1] if wins else None
        if last and last["kind"] == "reach" and t0 <= last["t1"] and max(t1, last["t1"]) - last["t0"] <= MAX_WINDOW:
            last["t1"], last["flagged"] = max(t1, last["t1"]), last["flagged"] or flagged
            last["reaches"].append(i)
        else:
            wins.append({"kind": "reach", "t0": t0, "t1": t1, "flagged": flagged, "reaches": [i]})

    reach_wins = list(wins)
    for m in moments:
        if m["track_id"] != tid or m["event"] != "hand_to_body":
            continue
        if any(w["t0"] <= m["t"] <= w["t1"] for w in wins):
            continue  # already in a window the llm will see
        t0, t1 = round(max(0.0, m["t"] - 2.0), 2), round(m["t"] + 2.0, 2)
        if wins[-1:] and wins[-1]["kind"] == "body" and t0 <= wins[-1]["t1"]:
            wins[-1]["t1"] = t1
        else:
            wins.append({"kind": "body", "t0": t0, "t1": t1, "flagged": True, "reaches": []})
    body_wins = [w for w in wins if w["kind"] == "body"]
    return ([w for w in reach_wins if w["flagged"]] + body_wins + [w for w in reach_wins if not w["flagged"]])


def followed_by_hand_to_body(moments, i, max_len=10.0):
    """did this person move a hand to their body after this reach, before their next real reach"""
    reach = moments[i]
    for m in moments[i + 1:]:
        if m["track_id"] != reach["track_id"]:
            continue
        if m["t"] - reach["t"] > max_len or (m["event"] == "hand_reach_shelf" and m.get("llm_call")):
            return False
        if m["event"] == "hand_to_body":
            return True
    return False


def _normalize(response):
    """llms dont always follow the format (gemini sometimes wraps the answer in a list). returns a dict or None."""
    if isinstance(response, list):
        response = next((r for r in response if isinstance(r, dict)), None)
    if not isinstance(response, dict):
        return None
    if not isinstance(response.get("claims", []), list):
        response = {**response, "claims": []}
    return response


def inspect_reaches(state):
    """check 1 + 2: is the view clear, and what happened to items. one llm call per window.

    first look checks every window. second look only re-checks the windows that werent clear,
    a few seconds longer and with more frames.
    """
    llm = state.get("llm")
    attempt = state["attempt"] + 1
    todo = state["windows"] if attempt == 1 else state.get("unclear", [])

    evidence, dropped = list(state["evidence"]), list(state["dropped"])
    calls, any_clear = state["llm_calls"], state["clear_view"]
    unclear = []

    for w in todo:
        if llm is None or calls >= state.get("max_calls", 3):
            break
        t0, t1 = w["t0"], w["t1"] if attempt == 1 else w["t1"] + 3.0
        n = 5 if attempt == 1 else 7
        n = min(n + 3, max(n, round((t1 - t0) / 1.2)))  # long windows (rummaging) get a few more frames
        frames = get_keyframes(state["video"], t0, t1, n=n)
        times = [t for t, _ in frames]
        blur = state.get("blur") or (lambda f: f)
        calls += 1
        prompt = PROMPT.format(n=len(frames), times=times, when=WHEN[w["kind"]])
        try:
            response = llm.ask_with_images(prompt, [blur(f) for _, f in frames])
        except Exception as e:
            dropped.append({"claim": None, "what": str(e)[:200], "reason": "llm error"})
            continue

        response = _normalize(response)
        if response is None:
            dropped.append({"claim": None, "what": "llm answer was not in the expected format", "reason": "bad llm response"})
            unclear.append(w)
            continue

        kept, bad, clear = verify(response, times)
        # a second look at the same moment replaces nothing, but dont count the same action twice
        seen = {(e["claim"], e["frame_time"]) for e in evidence}
        evidence += [k for k in kept if (k["claim"], k["frame_time"]) not in seen]
        dropped += bad
        any_clear = any_clear or clear
        if not clear and not any(k["claim"] == "item_hidden" for k in kept):
            unclear.append(w)

    return {"attempt": attempt, "llm_calls": calls, "evidence": evidence, "dropped": dropped,
            "clear_view": any_clear, "unclear": unclear}


def need_more_data(state):
    # loop back once, only for the reaches we couldnt see clearly the first time
    can_call = state.get("llm") is not None and state["llm_calls"] < state.get("max_calls", 3)
    if state.get("unclear") and state["attempt"] < 2 and can_call:
        return "inspect_reaches"
    return "check_counter"


def check_counter(state):
    """check 3: did they visit the counter before leaving. no llm, just the timeline."""
    timeline = state["timeline"]
    exits = [m["t"] for m in timeline if m["event"] in EXIT_EVENTS]
    counters = [m["t"] for m in timeline if m["event"] == "at_counter"]
    # only this persons own events count, other people at the counter dont change their case
    exited = bool(exits)
    if counters:
        counter_visit = not exits or min(counters) < max(exits)
    else:
        counter_visit = False if exited else None  # None = still in the store / cant tell yet
    return {"counter_visit": counter_visit, "exited": exited}


def follow_items(evidence):
    """go through what the llm saw in time order and work out where the items ended up.

    returns (in_hand, hidden, ever_hidden, put_back): items still in the hand, items still hidden,
    whether anything was hidden at some point, how many went back on the shelf
    """
    in_hand = hidden = put_back = 0
    ever_hidden = False
    for e in sorted(evidence, key=lambda e: e.get("frame_time") or 0):
        claim = e["claim"]
        if claim == "item_taken":
            in_hand += 1
        elif claim == "item_hidden":
            hidden += 1
            ever_hidden = True
            in_hand = max(0, in_hand - 1)
        elif claim == "item_taken_out":
            if hidden:
                hidden -= 1
            in_hand += 1
        elif claim == "item_put_back":
            put_back += 1
            if in_hand:
                in_hand -= 1
            elif hidden:  # straight from the pocket back to the shelf
                hidden -= 1
    return in_hand, hidden, ever_hidden, put_back


def decide(state):
    in_hand, hidden, ever_hidden, put_back = follow_items(state["evidence"])
    counter, exited = state.get("counter_visit"), state.get("exited")
    left_without_paying = exited and counter is False

    if hidden and left_without_paying:
        severity = "L2"
        why = "item hidden and left without visiting the counter"
    elif hidden:
        severity = "L1"
        why = "item hidden" + (" but visited the counter" if counter else ", exit not seen")
    elif in_hand and left_without_paying:
        severity = "L1"
        why = "item taken and left without visiting the counter" + (
            " (it was hidden for a while, then taken out)" if ever_hidden else "")
    elif ever_hidden:
        severity = "normal"
        why = "item was hidden but taken out again" + (" and put back" if not in_hand else "")
    elif put_back and not in_hand:
        severity = "normal"
        why = "picked up item(s) and put them back"
    elif state["deviation"] and state["llm_calls"] == 0:
        severity = "inconclusive"
        why = "unusual visit but not checked (no llm calls left)"
        if state.get("flagged"):
            why = "hand moved to pocket/bag after the shelf (pose only), not checked by the llm yet"
    elif not state["clear_view"] and state["deviation"]:
        severity = "inconclusive"
        why = "hands not clearly visible, evidence obscured"
    else:
        severity = "normal"
        why = "nothing suspicious seen"

    evidence = sorted(state["evidence"], key=lambda e: e.get("frame_time") or 0)
    lines = [f"{e['frame_time']:.1f}s {e['claim']}: {clean_text(e['what'])}" for e in evidence]
    summary = why + (". " + " | ".join(lines) if lines else "")
    claims = {e["claim"] for e in evidence}
    checks = {"clear_view": state["clear_view"], "item_hidden": bool(hidden), "item_taken": bool(in_hand),
              "hidden_then_taken_out": ever_hidden and not hidden, "put_back": put_back,
              "claims_seen": sorted(claims),
              "hand_to_body": state.get("flagged", 0) > 0,
              "counter_visit": counter, "exited": exited, "looks": state["attempt"]}
    return {"severity": severity, "summary": summary, "checks": checks, "evidence": evidence}


def no_deviation(state):
    return {"severity": "normal", "summary": "normal visit (shelf, counter, exit) or no shelf reach",
            "checks": {}, "llm_calls": 0, "evidence": [], "dropped": []}


def build_graph():
    g = StateGraph(CaseState)
    g.add_node("build_timeline", build_timeline)
    g.add_node("no_deviation", no_deviation)
    g.add_node("inspect_reaches", inspect_reaches)
    g.add_node("check_counter", check_counter)
    g.add_node("decide", decide)

    g.add_edge(START, "build_timeline")
    g.add_conditional_edges("build_timeline", lambda s: "inspect_reaches" if s["deviation"] else "no_deviation",
                            ["inspect_reaches", "no_deviation"])
    g.add_edge("no_deviation", END)
    g.add_conditional_edges("inspect_reaches", need_more_data, ["inspect_reaches", "check_counter"])
    g.add_edge("check_counter", "decide")
    g.add_edge("decide", END)
    return g.compile()


def run_case(video, track_id, moments, llm=None, max_calls=3, blur=None, graph=None):
    graph = graph or build_graph()
    state = graph.invoke({"video": video, "track_id": track_id, "moments": moments, "llm": llm,
                          "max_calls": max_calls, "blur": blur})
    return {
        "video": video, "track_id": track_id, "severity": state["severity"], "summary": state["summary"],
        "checks": state.get("checks", {}), "evidence": state.get("evidence", []),
        "dropped": state.get("dropped", []), "llm_calls": state.get("llm_calls", 0),
    }
