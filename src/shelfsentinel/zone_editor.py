# fix zones by hand on the live zones screen: select, drag corners, move, redraw, delete, undo, cancel
#
# all coordinates here are camera-frame pixels. the live window maps mouse positions to frame pixels
# before calling on_mouse.

import copy

import cv2
import numpy as np

from shelfsentinel.perception.zones import point_in_polygon

NAMES = ("shelf", "counter", "exit")
KEY_TO_ZONE = {ord("1"): "shelf", ord("2"): "counter", ord("3"): "exit"}
HANDLE_RADIUS = 14


class ZoneEditor:
    def __init__(self, zones):
        self.load(zones)

    def load(self, zones):
        """new starting point (e.g. after gemini / yolo found zones again)"""
        self.zones = copy.deepcopy({k: [list(map(int, p)) for p in v] for k, v in zones.items()})
        self.original = copy.deepcopy(self.zones)
        self.history = []
        self.selected = None
        self.drawing = None  # list of points while redrawing the selected zone
        self.drag = None  # (name, vertex index or None for moving the whole zone, last point)

    # ---- helpers
    def _save(self):
        self.history.append(copy.deepcopy(self.zones))
        self.history = self.history[-50:]

    def _nearest_vertex(self, x, y):
        order = ([self.selected] if self.selected in self.zones else []) + [n for n in self.zones if n != self.selected]
        for name in order:
            for i, (vx, vy) in enumerate(self.zones[name]):
                if (vx - x) ** 2 + (vy - y) ** 2 <= HANDLE_RADIUS ** 2:
                    return name, i
        return None

    def _zone_at(self, x, y):
        for name in ([self.selected] if self.selected in self.zones else []) + list(self.zones):
            if point_in_polygon((x, y), self.zones[name]):
                return name
        return None

    # ---- input
    def on_mouse(self, event, x, y):
        x, y = int(x), int(y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing is not None:
                self.drawing.append([x, y])
                return
            hit = self._nearest_vertex(x, y)
            if hit:
                self._save()
                self.selected = hit[0]
                self.drag = (hit[0], hit[1], (x, y))
                return
            name = self._zone_at(x, y)
            self.selected = name
            if name:
                self._save()
                self.drag = (name, None, (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            name, index, (lx, ly) = self.drag
            if index is None:  # move the whole zone
                self.zones[name] = [[px + x - lx, py + y - ly] for px, py in self.zones[name]]
            else:
                self.zones[name][index] = [x, y]
            self.drag = (name, index, (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def on_key(self, key):
        """returns True if the editor used the key"""
        if self.drawing is not None:
            if key in (13, 10):  # enter: finish the new zone
                if len(self.drawing) >= 3:
                    self._save()
                    self.zones[self.selected] = self.drawing
                self.drawing = None
                return True
            if key == 27:  # esc: cancel the redraw
                self.drawing = None
                return True
            if key == 8 and self.drawing:  # backspace: remove last point
                self.drawing.pop()
                return True
            return True  # ignore other keys while drawing
        if key in KEY_TO_ZONE:
            self.selected = KEY_TO_ZONE[key]
            return True
        if key == ord("r") and self.selected:
            self.drawing = []
            return True
        if key == ord("d") and self.selected in self.zones:
            self._save()
            del self.zones[self.selected]
            return True
        if key == ord("u") and self.history:
            self.zones = self.history.pop()
            return True
        if key == ord("c"):
            self.zones = copy.deepcopy(self.original)
            self.history, self.drawing, self.drag = [], None, None
            return True
        return False

    # ---- drawing + help
    def draw_handles(self, img, color=(255, 255, 255)):
        if self.selected in self.zones:
            pts = np.array(self.zones[self.selected], np.int32)
            cv2.polylines(img, [pts], True, color, 3, cv2.LINE_AA)
            for px, py in self.zones[self.selected]:
                cv2.circle(img, (int(px), int(py)), 9, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 9, (0, 0, 0), 2, cv2.LINE_AA)
        if self.drawing is not None:
            for i, p in enumerate(self.drawing):
                cv2.circle(img, tuple(p), 7, (255, 255, 0), -1, cv2.LINE_AA)
                if i:
                    cv2.line(img, tuple(self.drawing[i - 1]), tuple(p), (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def help_lines(self):
        if self.drawing is not None:
            return [f"drawing {self.selected.upper()}",
                    "click the corners",
                    "enter  finish",
                    "backspace  remove last point",
                    "esc  cancel"]
        sel = self.selected.upper() if self.selected else "none"
        missing = [n for n in NAMES if n not in self.zones]
        lines = [f"selected: {sel}",
                 "1 shelf   2 counter   3 exit",
                 "click a zone to select it",
                 "drag a corner / drag inside to move",
                 "r redraw   d delete   u undo",
                 "c cancel all edits"]
        if missing:
            lines.append("missing: " + ", ".join(missing) + " (select + r)")
        return lines
