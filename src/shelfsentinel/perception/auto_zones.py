# find the shelf, counter and exit zones automatically from one camera frame, so we dont have to
# redraw them every time the camera or the room changes.
#
# two ways:
#   gemini: one llm call, asks for the boxes of the shelf / counter / exit (most accurate, needs internet + 1 call)
#   yolo:   yolo-world on the laptop, looks for "shelf", "counter", "door"... (free, offline, less reliable)
#
# a zone is a polygon in pixels, same format as zones.yaml: {"shelf": [[x, y], ...], "counter": [...], "exit": [...]}
#
# shelf zone = the shelf itself (the event detector checks the wrists against it)
# counter / exit zones = the floor in front of them, because the event detector checks where the feet are

ZONE_NAMES = ("shelf", "counter", "exit")

GEMINI_PROMPT = (
    "This is a frame from a fixed camera in a small store (it may be a mock store made of tables and boxes). "
    "Find these three areas: "
    "shelf = the shelf or display where customers pick up items (the main one if there are several), "
    "counter = the checkout counter / register / table where people pay, "
    "exit = the door or doorway people leave through (if no door is visible, the edge of the frame people walk out of). "
    "Faces are blurred on purpose. "
    "Answer as JSON with a box for each, as [ymin, xmin, ymax, xmax] scaled 0 to 1000, or null if you cant see it: "
    '{"shelf": [ymin, xmin, ymax, xmax] or null, "counter": [...] or null, "exit": [...] or null}'
)

# what yolo-world looks for, and which zone each word belongs to
YOLO_WORDS = {
    "shelf": "shelf", "store shelf": "shelf", "shelving unit": "shelf", "display rack": "shelf",
    "bookshelf": "shelf",
    "checkout counter": "counter", "cash register": "counter", "counter": "counter", "table": "counter",
    "desk": "counter",
    "door": "exit", "doorway": "exit", "exit sign": "exit",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box_to_zone(name, box, width, height):
    """turn a box (x1, y1, x2, y2 in pixels) into the zone polygon the event detector needs"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if name == "counter":
        # where someone stands to pay: from the lower part of the counter down to the floor in front of it
        x1, x2 = x1 - 0.15 * w, x2 + 0.15 * w
        y1, y2 = y2 - 0.3 * h, y2 + 0.6 * h
    elif name == "exit":
        # around the bottom of the door, where the feet are when someone walks out
        x1, x2 = x1 - 0.1 * w, x2 + 0.1 * w
        y1, y2 = y2 - 0.35 * h, y2 + 0.25 * h
    x1, x2 = _clamp(x1, 0, width - 1), _clamp(x2, 0, width - 1)
    y1, y2 = _clamp(y1, 0, height - 1), _clamp(y2, 0, height - 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]


def boxes_to_zones(boxes, width, height):
    """boxes: {name: (x1, y1, x2, y2) or None} in pixels -> zones dict, missing ones left out"""
    zones = {}
    for name in ZONE_NAMES:
        box = boxes.get(name)
        if box is None:
            continue
        zone = box_to_zone(name, box, width, height)
        if zone:
            zones[name] = zone
    return zones


def _gemini_box(value, width, height):
    """gemini gives [ymin, xmin, ymax, xmax] scaled 0-1000"""
    if isinstance(value, dict):
        value = value.get("box_2d") or value.get("box")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin / 1000 * width, ymin / 1000 * height, xmax / 1000 * width, ymax / 1000 * height)


def find_zones_gemini(frame, llm, blur=None):
    """one llm call. returns (zones, raw answer)"""
    height, width = frame.shape[:2]
    image = blur(frame) if blur else frame
    answer = llm.ask_with_images(GEMINI_PROMPT, [image])
    if isinstance(answer, list):
        answer = next((a for a in answer if isinstance(a, dict)), {})
    if not isinstance(answer, dict):
        answer = {}
    boxes = {name: _gemini_box(answer.get(name), width, height) for name in ZONE_NAMES}
    return boxes_to_zones(boxes, width, height), answer


_yolo_world = None


def _get_yolo_world(weights="yolov8s-worldv2.pt"):
    global _yolo_world
    if _yolo_world is None:
        from ultralytics import YOLOWorld

        _yolo_world = YOLOWorld(weights)
        _yolo_world.set_classes(list(YOLO_WORDS))
    return _yolo_world


def pick_yolo_boxes(detections):
    """detections: [(word, confidence, (x1, y1, x2, y2))] -> best box per zone (confidence x size)"""
    best = {}
    for word, conf, box in detections:
        name = YOLO_WORDS.get(word)
        if name is None:
            continue
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        score = conf * area ** 0.5  # prefer confident and big (the main shelf, not a small box)
        if name not in best or score > best[name][0]:
            best[name] = (score, box)
    return {name: box for name, (_, box) in best.items()}


def find_zones_yolo(frame, model=None, conf=0.05):
    """yolo-world on the laptop, free and offline. returns (zones, detections)"""
    height, width = frame.shape[:2]
    model = model or _get_yolo_world()
    result = model.predict(frame, conf=conf, verbose=False)[0]
    detections = [
        (result.names[int(c)], float(s), tuple(float(v) for v in b))
        for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)
    ]
    return boxes_to_zones(pick_yolo_boxes(detections), width, height), detections
