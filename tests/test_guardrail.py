import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shelfsentinel.guardrail.verify import clean_text, verify

TIMES = [10.0, 11.0, 12.0]


def test_keeps_claim_with_frame():
    kept, dropped, clear = verify({"clear_view": True, "claims": [
        {"claim": "item_hidden", "frame": 3, "what": "puts the box into the bag"}]}, TIMES)
    assert clear and dropped == []
    assert kept[0]["frame_time"] == 12.0


def test_drops_missing_or_bad_frame():
    kept, dropped, _ = verify({"clear_view": True, "claims": [
        {"claim": "item_taken", "what": "takes a box"},
        {"claim": "item_taken", "frame": 7, "what": "takes a box"}]}, TIMES)
    assert kept == [] and len(dropped) == 2


def test_drops_hedged_claim():
    kept, dropped, _ = verify({"clear_view": True, "claims": [
        {"claim": "item_hidden", "frame": 2, "what": "appears to place it in a pocket"}]}, TIMES)
    assert kept == [] and dropped[0]["reason"] == "hedged, not seen"


def test_hidden_needs_clear_view():
    kept, dropped, clear = verify({"clear_view": False, "claims": [
        {"claim": "item_taken", "frame": 1, "what": "takes a box"},
        {"claim": "item_hidden", "frame": 2, "what": "puts it in a pocket"}]}, TIMES)
    assert not clear
    assert [k["claim"] for k in kept] == ["item_taken"]
    assert dropped[0]["reason"] == "view not clear"


def test_unknown_claim_dropped():
    kept, dropped, _ = verify({"clear_view": True, "claims": [{"claim": "is_a_thief", "frame": 1, "what": "x"}]}, TIMES)
    assert kept == [] and dropped[0]["reason"] == "unknown claim"


def test_clean_text_removes_appearance():
    out = clean_text("The woman, wearing a grey jacket, puts a white box in her bag")
    assert "woman" not in out and "jacket" not in out
    assert "white box" in out


def test_text_booleans_and_frames_are_read_correctly():
    kept, dropped, clear = verify({"clear_view": "false", "claims": [
        {"claim": "item_hidden", "frame": "2", "what": "puts it in a pocket"}]}, TIMES)
    assert clear is False
    assert kept == [] and dropped[0]["reason"] == "view not clear"
