# drawing for the live window: camera on the left, a side panel with status, events and the latest case

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

# BGR
BG = (24, 24, 28)
CARD = (40, 40, 46)
TEXT = (235, 235, 235)
MUTED = (150, 150, 155)
RED = (60, 60, 230)
ORANGE = (40, 150, 245)
GREEN = (90, 200, 90)
YELLOW = (40, 200, 240)
GRAY = (130, 130, 130)
BLUE = (230, 150, 60)

ZONE_COLORS = {"shelf": YELLOW, "counter": GREEN, "exit": RED}
ZONE_LABELS = {"shelf": "SHELF", "counter": "COUNTER", "exit": "EXIT"}

EVENT_STYLE = {
    "hand_reach_shelf": ("hand at shelf", YELLOW),
    "hand_to_body": ("hand to pocket / bag", RED),
    "at_counter": ("at the counter", GREEN),
    "leave_counter": ("left the counter", GREEN),
    "person_exited": ("left the store", MUTED),
}
SEVERITY_STYLE = {
    "L2": ("L2  POSSIBLE THEFT", RED),
    "L1": ("L1  UNUSUAL VISIT", ORANGE),
    "inconclusive": ("INCONCLUSIVE", GRAY),
    "normal": ("NORMAL", GREEN),
}

PANEL_W = 400
VIEW_H = 720


def text(img, s, org, scale=0.6, color=TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, AA)


def pill(img, s, org, color, scale=0.55, fg=(20, 20, 20)):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - 8), (x + w + 16, y + 6), color, -1, AA)
    text(img, s, (x + 8, y), scale, fg)
    return x + w + 16


def wrap(s, width_px, scale=0.5):
    words, lines, line = str(s).split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if cv2.getTextSize(test, FONT, scale, 1)[0][0] > width_px and line:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def draw_zones(img, zones, alpha=0.18):
    overlay = img.copy()
    for name, polygon in zones.items():
        pts = np.array([[int(x), int(y)] for x, y in polygon], np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLORS.get(name, GRAY))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    for name, polygon in zones.items():
        pts = np.array([[int(x), int(y)] for x, y in polygon], np.int32)
        color = ZONE_COLORS.get(name, GRAY)
        cv2.polylines(img, [pts], True, color, 2, AA)
        top_left = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        pill(img, ZONE_LABELS.get(name, name.upper()), (int(top_left[0]) + 4, int(top_left[1]) + 24), color, 0.5)
    return img


def draw_people(img, result):
    """skeletons from yolo, plus a small #id tag instead of the big default label"""
    if result is None:
        return img
    img = result.plot(img=img, line_width=2, labels=False, conf=False, boxes=False)
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        for tid, (x1, y1, x2, y2) in zip(boxes.id.int().tolist(), boxes.xyxy.tolist()):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BLUE, 2, AA)
            pill(img, f"#{tid}", (int(x1), max(22, int(y1) - 4)), BLUE, 0.55, (255, 255, 255))
    return img


def fit(img, height=VIEW_H):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)


def panel(height, status, timer, events, case, zones_source, hint, help_lines=None):
    p = np.full((height, PANEL_W, 3), BG, np.uint8)
    text(p, "ShelfSentinel", (24, 46), 0.95, TEXT, 2)
    pill(p, "LIVE", (300, 44), RED, 0.55, (255, 255, 255))

    color = {"RECORDING": RED, "CHECKING": ORANGE, "WAITING": GRAY, "SETUP": BLUE}.get(status.split()[0], GRAY)
    end = pill(p, status, (24, 100), color, 0.6, (255, 255, 255))
    if timer:
        text(p, timer, (end + 12, 100), 0.6, TEXT)
    text(p, f"zones: {zones_source}", (24, 132), 0.45, MUTED)

    cv2.line(p, (24, 152), (PANEL_W - 24, 152), CARD, 1)
    if help_lines:  # zones screen: show how to edit instead of the events
        text(p, "EDIT ZONES", (24, 182), 0.5, MUTED)
        for i, line in enumerate(help_lines):
            text(p, line, (24, 216 + 30 * i), 0.52, YELLOW if i == 0 else TEXT)
        text(p, hint, (24, height - 30), 0.48, MUTED)
        return p
    text(p, "EVENTS", (24, 182), 0.5, MUTED)
    y = 214
    for t, track, name in events[-9:]:
        label, c = EVENT_STYLE.get(name, (name, MUTED))
        cv2.circle(p, (32, y - 6), 6, c, -1, AA)
        text(p, f"{t:5.1f}s", (48, y), 0.5, MUTED)
        text(p, f"#{track} {label}", (118, y), 0.52, TEXT)
        y += 32
    if not events:
        text(p, "nothing yet", (24, y), 0.5, MUTED)

    card_top = height - 230
    text(p, "LATEST CASE", (24, card_top - 14), 0.5, MUTED)
    cv2.rectangle(p, (24, card_top), (PANEL_W - 24, height - 70), CARD, -1, AA)
    if case:
        label, c = SEVERITY_STYLE.get(case["severity"], (case["severity"], GRAY))
        cv2.rectangle(p, (24, card_top), (32, height - 70), c, -1)
        text(p, label, (46, card_top + 32), 0.62, c, 2)
        text(p, f"track #{case['track_id']}", (46, card_top + 58), 0.45, MUTED)
        for i, line in enumerate(wrap(case["summary"], PANEL_W - 90)[:3]):
            text(p, line, (46, card_top + 88 + 24 * i), 0.48, TEXT)
    else:
        text(p, "cases show up here after a visit", (46, card_top + 40), 0.48, MUTED)

    text(p, hint, (24, height - 30), 0.48, MUTED)
    return p


def banner(img, lines):
    """big dark bar at the top of the camera view with instructions"""
    pad, line_h = 14, 40
    h = pad * 2 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, line in enumerate(lines):
        text(img, line, (pad + 6, pad + 28 + line_h * i), 0.8, YELLOW if i == 0 else TEXT, 2)
    return img


def compose(camera_img, zones, result, status, timer, events, case, zones_source, hint, editor=None,
            banner_lines=None):
    view = camera_img.copy()
    view = draw_people(view, result)
    view = draw_zones(view, zones)
    if editor is not None:
        view = editor.draw_handles(view)
    view = fit(view)
    if banner_lines:
        view = banner(view, banner_lines)
    help_lines = editor.help_lines() if editor is not None else None
    return np.hstack([view, panel(view.shape[0], status, timer, events, case, zones_source, hint, help_lines)])


def view_scale(camera_img):
    """how much the camera image is scaled in the window (to map mouse clicks back to camera pixels)"""
    h, w = camera_img.shape[:2]
    return VIEW_H / h, int(w * VIEW_H / h)
