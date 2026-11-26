"""auto zones with fake gemini / fake yolo-world, no downloads or api calls"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.perception.auto_zones import (
    box_to_zone,
    find_zones_gemini,
    find_zones_yolo,
    pick_yolo_boxes,
)
from shelfsentinel.perception.zones import point_in_polygon

FRAME = np.zeros((500, 1000, 3), np.uint8)  # 1000 wide, 500 high


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def ask_with_images(self, prompt, frames):
        self.calls += 1
        return self.answer


def test_gemini_boxes_become_zones():
    llm = FakeLLM({"shelf": [100, 0, 600, 300], "counter": [400, 600, 700, 900], "exit": None})
    zones, _ = find_zones_gemini(FRAME, llm)
    assert llm.calls == 1
    assert set(zones) == {"shelf", "counter"}
    # shelf: 0-300 wide of 1000 px, 100-600 of 1000 high -> 0..300 x 50..300 y
    assert zones["shelf"] == [[0, 50], [300, 50], [300, 300], [0, 300]]


def test_gemini_answer_in_a_list_or_as_box_2d():
    llm = FakeLLM([{"shelf": {"box_2d": [0, 0, 500, 500]}, "counter": "nope", "exit": [1, 2, 3]}])
    zones, _ = find_zones_gemini(FRAME, llm)
    assert set(zones) == {"shelf"}


def test_counter_zone_is_the_floor_in_front_of_it():
    zone = box_to_zone("counter", (400, 100, 600, 200), 1000, 500)
    # feet of someone standing in front of the counter (below it in the image) count
    assert point_in_polygon((500, 240), zone)
    # the top of the counter itself does not
    assert not point_in_polygon((500, 110), zone)


def test_exit_zone_is_around_the_bottom_of_the_door():
    zone = box_to_zone("exit", (800, 50, 900, 350), 1000, 500)
    assert point_in_polygon((850, 340), zone)
    assert not point_in_polygon((850, 80), zone)


def test_zones_stay_inside_the_frame():
    zone = box_to_zone("counter", (900, 400, 1000, 500), 1000, 500)
    assert all(0 <= x < 1000 and 0 <= y < 500 for x, y in zone)


def test_pick_yolo_boxes_prefers_confident_big_boxes_and_maps_words():
    picked = pick_yolo_boxes([
        ("shelf", 0.3, (0, 0, 50, 50)),
        ("store shelf", 0.25, (0, 0, 400, 400)),
        ("table", 0.4, (500, 300, 700, 400)),
        ("door", 0.2, (900, 0, 1000, 400)),
        ("person", 0.9, (0, 0, 10, 10)),
    ])
    assert picked["shelf"] == (0, 0, 400, 400)
    assert picked["counter"] == (500, 300, 700, 400)
    assert picked["exit"] == (900, 0, 1000, 400)


class FakeBoxes:
    def __init__(self, rows):
        self.cls = [r[0] for r in rows]
        self.conf = [r[1] for r in rows]
        self.xyxy = [r[2] for r in rows]


class FakeYolo:
    names = {0: "shelf", 1: "checkout counter", 2: "doorway"}

    def predict(self, frame, conf, verbose):
        result = type("R", (), {})()
        result.names = self.names
        result.boxes = FakeBoxes([(0, 0.5, (10, 10, 300, 400)), (1, 0.4, (600, 300, 800, 380)),
                                  (2, 0.3, (900, 20, 990, 450))])
        return [result]


def test_find_zones_yolo_with_a_fake_model():
    zones, detections = find_zones_yolo(FRAME, model=FakeYolo())
    assert set(zones) == {"shelf", "counter", "exit"}
    assert len(detections) == 3
