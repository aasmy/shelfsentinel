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


LIVE_BUTTONS = [("end_visit", "END VISIT", ORANGE), ("edit_zones", "EDIT ZONES", BLUE), ("stop", "STOP", RED)]
SETUP_BUTTONS = [("start", "START", GREEN), ("quit", "QUIT", RED)]


def button_rects(buttons, height, x0=0):
    """where each button is, in panel coords (or canvas coords with x0 = camera view width)"""
    gap, top, bottom = 10, height - 120, height - 70
    width = (PANEL_W - 48 - gap * (len(buttons) - 1)) / len(buttons)
    rects = {}
    for i, (name, _, _) in enumerate(buttons):
        x1 = int(x0 + 24 + i * (width + gap))
        rects[name] = (x1, top, int(x1 + width), bottom)
    return rects


def draw_buttons(p, buttons):
    for (name, label, color), (x1, y1, x2, y2) in zip(buttons, button_rects(buttons, p.shape[0]).values()):
        cv2.rectangle(p, (x1, y1), (x2, y2), color, -1, AA)
        (w, h), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        text(p, label, (x1 + (x2 - x1 - w) // 2, y1 + (y2 - y1 + h) // 2), 0.55, (255, 255, 255), 1)


def panel(height, status, timer, events, case, zones_source, hint, help_lines=None, buttons=None):
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
        if buttons:
            draw_buttons(p, buttons)
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

    card_top, card_bottom = height - 320, height - 140
    text(p, "LATEST CASE", (24, card_top - 14), 0.5, MUTED)
    cv2.rectangle(p, (24, card_top), (PANEL_W - 24, card_bottom), CARD, -1, AA)
    if case:
        label, c = SEVERITY_STYLE.get(case["severity"], (case["severity"], GRAY))
        cv2.rectangle(p, (24, card_top), (32, card_bottom), c, -1)
        text(p, label, (46, card_top + 32), 0.62, c, 2)
        text(p, f"track #{case['track_id']}", (46, card_top + 58), 0.45, MUTED)
        for i, line in enumerate(wrap(case["summary"], PANEL_W - 90)[:4]):
            text(p, line, (46, card_top + 88 + 24 * i), 0.48, TEXT)
    else:
        text(p, "cases show up here after a visit", (46, card_top + 40), 0.48, MUTED)

    if buttons:
        draw_buttons(p, buttons)
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
            banner_lines=None, buttons=None):
    view = camera_img.copy()
    view = draw_people(view, result)
    view = draw_zones(view, zones)
    if editor is not None:
        view = editor.draw_handles(view)
    view = fit(view)
    if banner_lines:
        view = banner(view, banner_lines)
    help_lines = editor.help_lines() if editor is not None else None
    return np.hstack([view, panel(view.shape[0], status, timer, events, case, zones_source, hint, help_lines,
                                  buttons)])


def clicked_button(buttons, view_w, x, y, height=VIEW_H):
    """name of the button at canvas position x, y, or None"""
    for name, (x1, y1, x2, y2) in button_rects(buttons, height, x0=view_w).items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def view_scale(camera_img):
    """how much the camera image is scaled in the window (to map mouse clicks back to camera pixels)"""
    h, w = camera_img.shape[:2]
    return VIEW_H / h, int(w * VIEW_H / h)


def window_to_image(window, x, y, image_size):
    """windows can show the window smaller than the picture (display scaling) and then the mouse
    positions come in window pixels. map them back to picture pixels."""
    try:
        _, _, shown_w, shown_h = cv2.getWindowImageRect(window)
    except cv2.error:
        return x, y
    w, h = image_size
    if shown_w <= 0 or shown_h <= 0 or (shown_w, shown_h) == (w, h):
        return x, y
    return int(x * w / shown_w), int(y * h / shown_h)
