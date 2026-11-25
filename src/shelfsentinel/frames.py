import cv2


def get_frames(video_path, t, offsets=(-1.0, 0.0, 1.0)):
    """Grab frames around time t (seconds). Returns a list of (time, frame)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    seen = set()
    for offset in offsets:
        idx = int(round((t + offset) * fps))
        idx = max(0, min(idx, total - 1))  # stay inside the video
        if idx in seen:  # can happen near the start or end of the video
            continue
        seen.add(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((round(idx / fps, 2), frame))

    cap.release()
    return frames


def get_keyframes(video_path, t0, t1, n=5):
    """Grab n evenly spaced frames between t0 and t1 (seconds). Returns a list of (time, frame)."""
    if n < 2 or t1 <= t0:
        return get_frames(video_path, t0, offsets=(0.0,))
    step = (t1 - t0) / (n - 1)
    return get_frames(video_path, t0, offsets=[i * step for i in range(n)])


def reach_window(moments, i, after=3.0, max_len=10.0):
    """Time window for a reach: from just before the reach to a few seconds after the hand left the shelf.

    Hiding an item usually happens after the hand leaves the shelf, so frames around the reach alone miss it.
    Uses the last hand_leaves_shelf of the same person before their next real reach (wrists flicker on the
    zone edge, so the first leave is often not the real one).
    """
    reach = moments[i]
    t0 = max(0.0, reach["t"] - 0.5)
    leave_t = reach["t"]
    for m in moments[i + 1:]:
        if m["track_id"] != reach["track_id"]:
            continue
        if m["t"] - reach["t"] > max_len:
            break
        if m["event"] == "hand_reach_shelf" and m.get("llm_call"):
            break  # next real reach, stop here
        if m["event"] == "hand_leaves_shelf":
            leave_t = m["t"]
    t1 = min(leave_t + after, t0 + max_len)
    return round(t0, 2), round(t1, 2)
