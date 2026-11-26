# case agent: one visit (track) in, one case out
#
# build_timeline -> inspect_reaches -> (view not clear? look again with more frames) -> check_counter -> decide
#
# only visits that dont follow reach -> counter -> exit get sent to the llm, normal visits cost 0 calls

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from shelfsentinel.frames import get_keyframes, reach_window
from shelfsentinel.guardrail.verify import clean_text, verify

EXIT_EVENTS = {"person_exited", "exit"}

PROMPT = (
    "These are {n} frames from a store camera, in order, numbered 1 to {n}, taken at {times} seconds. "
    "They start when a person reaches into the shelf and continue after their hand leaves it. "
    "Faces are blurred on purpose. Only describe what the hands do with items. "
    "Do not describe the person. Only report what you can actually see, not what might have happened. "
    "Answer as JSON: "
    '{{"clear_view": true/false (can you see the hands well enough to tell?), '
    '"claims": [{{"claim": "item_taken" | "item_put_back" | "item_hidden", '
    '"frame": number of the frame that shows it, "what": "short description"}}]}}. '
    "Use an empty claims list if nothing happened."
)


class CaseState(TypedDict, total=False):
    video: str
    track_id: int
    moments: list  # all events from the video, in time order
    llm: object  # anything with ask_with_images(prompt, frames) -> dict, or None
    max_calls: int
    blur: object  # function frame -> frame

    timeline: list
    reaches: list  # indexes into moments of the reaches worth checking
    unclear: list  # reaches the llm couldnt see clearly, checked again on the second look
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

    return {"timeline": timeline, "reaches": reaches, "deviation": bool(reaches) and not normal,
            "attempt": 0, "llm_calls": 0, "evidence": [], "dropped": [], "clear_view": False}


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
    """check 1 + 2: is the view clear, and did an item get taken / hidden. one llm call per reach.

    first look checks every reach. second look only re-checks the reaches that werent clear,
    with a wider window and more frames.
    """
    llm = state.get("llm")
    attempt = state["attempt"] + 1
    after = 3.0 if attempt == 1 else 6.0
    n = 5 if attempt == 1 else 7
    todo = state["reaches"] if attempt == 1 else state.get("unclear", [])

    evidence, dropped = list(state["evidence"]), list(state["dropped"])
    calls, any_clear = state["llm_calls"], state["clear_view"]
    unclear = []

    for i in todo:
        if llm is None or calls >= state.get("max_calls", 3):
            break
        t0, t1 = reach_window(state["moments"], i, after=after)
        frames = get_keyframes(state["video"], t0, t1, n=n)
        times = [t for t, _ in frames]
        blur = state.get("blur") or (lambda f: f)
        calls += 1
        try:
            response = llm.ask_with_images(PROMPT.format(n=len(frames), times=times), [blur(f) for _, f in frames])
        except Exception as e:
            dropped.append({"claim": None, "what": str(e)[:200], "reason": "llm error"})
            continue

        response = _normalize(response)
        if response is None:
            dropped.append({"claim": None, "what": "llm answer was not in the expected format", "reason": "bad llm response"})
            unclear.append(i)
            continue

        kept, bad, clear = verify(response, times)
        evidence += kept
        dropped += bad
        any_clear = any_clear or clear
        if not clear and not any(k["claim"] == "item_hidden" for k in kept):
            unclear.append(i)

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


def decide(state):
    claims = {e["claim"] for e in state["evidence"]}
    hidden = "item_hidden" in claims
    taken = "item_taken" in claims
    counter, exited = state.get("counter_visit"), state.get("exited")

    if hidden and exited and counter is False:
        severity = "L2"
        why = "item hidden and left without visiting the counter"
    elif hidden:
        severity = "L1"
        why = "item hidden" + (" but visited the counter" if counter else ", exit not seen")
    elif taken and exited and counter is False:
        severity = "L1"
        why = "item taken and left without visiting the counter"
    elif state["deviation"] and state["llm_calls"] == 0:
        severity = "inconclusive"
        why = "unusual visit but not checked (no llm calls left)"
    elif not state["clear_view"] and state["deviation"]:
        severity = "inconclusive"
        why = "hands not clearly visible, evidence obscured"
    else:
        severity = "normal"
        why = "nothing suspicious seen"

    lines = [f"{e['frame_time']:.1f}s {e['claim']}: {clean_text(e['what'])}" for e in state["evidence"]]
    summary = why + (". " + " | ".join(lines) if lines else "")
    checks = {"clear_view": state["clear_view"], "item_hidden": hidden, "item_taken": taken,
              "counter_visit": counter, "exited": exited, "looks": state["attempt"]}
    return {"severity": severity, "summary": summary, "checks": checks}


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
