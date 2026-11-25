# guardrail: every claim the llm makes must point at a frame we actually sent, or it gets dropped

import re

ALLOWED_CLAIMS = {"item_taken", "item_put_back", "item_hidden"}

# llm is guessing, not seeing
HEDGE_WORDS = re.compile(r"\b(appears?|appearing|seems?|seemingly|presumably|likely|possibly|probably|suggest\w*|may|might|could)\b", re.I)

# behaviour only: no words about who the person is (proposal: appearance words removed)
PERSON_WORDS = re.compile(r"\b(man|woman|men|women|girl|boy|lady|guy|male|female)\b", re.I)
APPEARANCE = re.compile(r"[^.,;]*\b(wearing|dressed|skin|hair|beard|ethnic\w*|elderly)\b[^.,;]*[.,;]?", re.I)


def clean_text(text):
    text = APPEARANCE.sub("", text or "")
    text = PERSON_WORDS.sub("person", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def verify(response, frame_times):
    """Check one llm answer against the frames that were sent.

    response: {"clear_view": bool, "claims": [{"claim": str, "frame": int (1 based), "what": str}], ...}
    frame_times: times (s) of the frames we sent, in order
    returns (kept, dropped, clear_view)
    """
    clear_view = bool(response.get("clear_view", False))
    kept, dropped = [], []

    for c in response.get("claims", []) or []:
        claim = c.get("claim")
        frame = c.get("frame")
        what = clean_text(c.get("what", ""))
        entry = {"claim": claim, "what": what}

        if claim not in ALLOWED_CLAIMS:
            dropped.append({**entry, "reason": "unknown claim"})
        elif not isinstance(frame, int) or not 1 <= frame <= len(frame_times):
            dropped.append({**entry, "reason": "no valid frame cited"})
        elif HEDGE_WORDS.search(c.get("what", "")):
            dropped.append({**entry, "frame_time": frame_times[frame - 1], "reason": "hedged, not seen"})
        elif claim == "item_hidden" and not clear_view:
            dropped.append({**entry, "frame_time": frame_times[frame - 1], "reason": "view not clear"})
        else:
            kept.append({**entry, "frame_time": frame_times[frame - 1]})

    return kept, dropped, clear_view
