# blur faces before any frame leaves the laptop (proposal: no faces sent to the cloud llm)

import cv2

FACE_POINTS = [0, 1, 2, 3, 4]  # nose, eyes, ears in coco keypoints

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO("yolo11n-pose.pt")
    return _model


def _blur_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    k = max(15, (min(x2 - x1, y2 - y1) // 2) * 2 + 1)  # odd kernel, big enough to hide the face
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def blur_faces(frame, conf=0.3):
    """Returns a copy of the frame with every detected person's head blurred."""
    out = frame.copy()
    result = _get_model().predict(out, classes=[0], conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes.xyxy.tolist()
    kp_xy = result.keypoints.xy.tolist() if result.keypoints is not None else None
    kp_conf = result.keypoints.conf.tolist() if result.keypoints is not None and result.keypoints.conf is not None else None

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        size = 0.5 * (x2 - x1)  # head is roughly half the body box width
        face = []
        if kp_xy is not None and kp_conf is not None:
            face = [kp_xy[i][k] for k in FACE_POINTS if kp_conf[i][k] >= conf]

        if face:
            cx = sum(p[0] for p in face) / len(face)
            cy = sum(p[1] for p in face) / len(face)
            _blur_box(out, cx - size / 2, cy - size * 0.6, cx + size / 2, cy + size * 0.5)
        else:
            # no face points (back turned or too small), just blur the top of the box to be safe
            _blur_box(out, x1, y1, x2, y1 + 0.22 * (y2 - y1))
    return out
