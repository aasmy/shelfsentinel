# baseline from the proposal: fixed frames spread over the whole visit, one llm call, no agent

import cv2

from shelfsentinel.frames import get_keyframes

PROMPT = (
    "These are {n} frames from a store camera, in order, taken at {times} seconds, covering a whole visit. "
    "Faces are blurred on purpose. Only judge what the hands do with items. "
    "Did a person hide an item (in a pocket, bag or clothing) instead of carrying it openly? "
    'Answer as JSON: {{"theft": true/false, "clear_view": true/false, "reason": "short reason"}}'
)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps


def run_baseline(video_path, llm, n=5, blur=None):
    """one call per video. returns {verdict, theft, clear_view, reason, llm_calls, frame_times}"""
    duration = video_duration(video_path)
    frames = get_keyframes(str(video_path), 0.0, max(duration - 0.1, 0.0), n=n)
    times = [t for t, _ in frames]
    blur = blur or (lambda f: f)
    result = {"frame_times": times, "llm_calls": 1, "theft": False, "clear_view": False, "reason": ""}
    try:
        answer = llm.ask_with_images(PROMPT.format(n=len(frames), times=times), [blur(f) for _, f in frames])
    except Exception as e:
        result["reason"] = f"llm error: {str(e)[:200]}"
        result["verdict"] = "inconclusive"
        return result

    if isinstance(answer, list):
        answer = next((a for a in answer if isinstance(a, dict)), {})
    if not isinstance(answer, dict):
        answer = {}
    result["theft"] = _as_bool(answer.get("theft", False))
    result["clear_view"] = _as_bool(answer.get("clear_view", False))
    result["reason"] = str(answer.get("reason", ""))[:300]
    if result["theft"]:
        result["verdict"] = "L2"
    elif not result["clear_view"]:
        result["verdict"] = "inconclusive"
    else:
        result["verdict"] = "normal"
    return result
