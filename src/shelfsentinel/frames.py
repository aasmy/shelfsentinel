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
