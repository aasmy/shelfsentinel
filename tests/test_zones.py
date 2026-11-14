import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shelfsentinel.perception.zones import point_in_polygon, get_zone

zones = {
    "shelf": [
        [100, 100],
        [300, 100],
        [300, 250],
        [100, 250]
    ],
    "counter": [
        [400, 100],
        [600, 100],
        [600, 250],
        [400, 250]
    ]
}

assert get_zone((200, 150), zones) == "shelf"
assert get_zone((500, 150), zones) == "counter"
assert get_zone((700, 150), zones) is None

print("All tests passed!")