"""zone editor logic without a window"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.zone_editor import ZoneEditor

ZONES = {"shelf": [[0, 0], [100, 0], [100, 100], [0, 100]],
         "counter": [[200, 200], [300, 200], [300, 300], [200, 300]]}
DOWN, MOVE, UP = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP


def test_drag_a_corner():
    e = ZoneEditor(ZONES)
    e.on_mouse(DOWN, 101, 99)
    e.on_mouse(MOVE, 150, 140)
    e.on_mouse(UP, 150, 140)
    assert e.selected == "shelf"
    assert e.zones["shelf"][2] == [150, 140]


def test_drag_inside_moves_the_whole_zone():
    e = ZoneEditor(ZONES)
    e.on_mouse(DOWN, 250, 250)
    e.on_mouse(MOVE, 260, 270)
    e.on_mouse(UP, 260, 270)
    assert e.zones["counter"][0] == [210, 220]


def test_delete_undo_and_cancel():
    e = ZoneEditor(ZONES)
    e.on_key(ord("2"))
    e.on_key(ord("d"))
    assert "counter" not in e.zones
    e.on_key(ord("u"))
    assert "counter" in e.zones
    e.on_mouse(DOWN, 0, 0)
    e.on_mouse(MOVE, 30, 30)
    e.on_key(ord("c"))
    assert e.zones == ZONES


def test_redraw_a_zone_and_add_a_missing_one():
    e = ZoneEditor(ZONES)
    e.on_key(ord("3"))  # exit doesnt exist yet
    e.on_key(ord("r"))
    for p in ((400, 0), (500, 0), (500, 80)):
        e.on_mouse(DOWN, *p)
    e.on_key(13)
    assert e.zones["exit"] == [[400, 0], [500, 0], [500, 80]]
    assert e.drawing is None


def test_redraw_needs_3_points_and_esc_cancels():
    e = ZoneEditor(ZONES)
    e.on_key(ord("1"))
    e.on_key(ord("r"))
    e.on_mouse(DOWN, 5, 5)
    e.on_key(13)  # only 1 point, keeps the old shelf
    assert e.zones["shelf"] == ZONES["shelf"]
    e.on_key(ord("r"))
    e.on_mouse(DOWN, 5, 5)
    e.on_key(27)
    assert e.zones["shelf"] == ZONES["shelf"] and e.drawing is None


def test_help_mentions_missing_zones():
    assert any("missing: exit" in line for line in ZoneEditor(ZONES).help_lines())
