"""Deterministic tests for the case agent; no detector, video, or API is used."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shelfsentinel.agent.graph as agent_graph
from shelfsentinel.agent.graph import build_graph, run_case
from shelfsentinel.storage.db import connect, list_cases, save_case


class FakeLLM:
    """Return queued responses or raise queued exceptions, recording every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def ask_with_images(self, prompt, frames):
        self.calls.append({"prompt": prompt, "frames": frames})
        if not self.responses:
            return {"clear_view": False, "claims": []}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def synthetic_frames(monkeypatch):
    """Replace OpenCV extraction with deterministic timestamped frame tokens."""
    windows = []

    def get_keyframes(video_path, t0, t1, n=5):
        windows.append({"video": video_path, "t0": t0, "t1": t1, "n": n})
        if n < 2:
            return [(round(t0, 2), f"frame-{len(windows)}-0")]
        return [
            (
                round(t0 + (t1 - t0) * i / (n - 1), 2),
                f"frame-{len(windows)}-{i}",
            )
            for i in range(n)
        ]

    monkeypatch.setattr(agent_graph, "get_keyframes", get_keyframes)
    return windows


def event(name, t, track_id=1, llm_call=False):
    return {"track_id": track_id, "event": name, "t": t, "llm_call": llm_call}


def hidden_response(frame=4):
    return {
        "clear_view": True,
        "claims": [
            {
                "claim": "item_hidden",
                "frame": frame,
                "what": "puts the item into a pocket",
            }
        ],
    }


def visit(track_id=1):
    return [
        event("hand_reach_shelf", 2.0, track_id, llm_call=True),
        event("hand_leaves_shelf", 3.0, track_id),
    ]


def run(moments, llm=None, **kwargs):
    return run_case(
        "sample.mp4",
        kwargs.pop("track_id", 1),
        moments,
        llm=llm,
        graph=build_graph(),
        blur=lambda frame: frame,
        **kwargs,
    )


def test_normal_shelf_counter_exit_visit_skips_llm(synthetic_frames):
    llm = FakeLLM([hidden_response()])
    moments = visit() + [
        event("at_counter", 8.0),
        event("person_exited", 12.0),
    ]

    case = run(moments, llm)

    assert case["severity"] == "normal"
    assert case["llm_calls"] == 0
    assert llm.calls == []
    assert synthetic_frames == []


def test_hidden_item_without_counter_and_with_exit_is_l2(synthetic_frames):
    llm = FakeLLM([hidden_response()])
    moments = visit() + [event("person_exited", 12.0)]

    case = run(moments, llm)

    assert case["severity"] == "L2"
    assert case["checks"]["counter_visit"] is False
    assert case["checks"]["exited"] is True
    assert case["llm_calls"] == len(llm.calls) == 1
    assert case["evidence"][0]["claim"] == "item_hidden"
    assert case["evidence"][0]["frame_time"] in [
        frame_time for frame_time, _ in synthetic_frame_times(synthetic_frames[0])
    ]


def synthetic_frame_times(window):
    """Recreate frame timestamps returned by the deterministic extractor."""
    if window["n"] < 2:
        return [(round(window["t0"], 2), None)]
    return [
        (
            round(
                window["t0"]
                + (window["t1"] - window["t0"]) * i / (window["n"] - 1),
                2,
            ),
            None,
        )
        for i in range(window["n"])
    ]


def test_first_inspection_uses_five_frames_and_retry_uses_seven(synthetic_frames):
    llm = FakeLLM(
        [
            {"clear_view": False, "claims": []},
            hidden_response(frame=7),
        ]
    )
    moments = visit() + [event("person_exited", 12.0)]

    case = run(moments, llm)

    assert case["severity"] == "L2"
    assert len(llm.calls) == case["llm_calls"] == 2
    assert [window["n"] for window in synthetic_frames] == [5, 7]
    assert synthetic_frames[0]["t0"] == synthetic_frames[1]["t0"] == 1.5
    assert synthetic_frames[0]["t1"] == 6.0
    assert synthetic_frames[1]["t1"] == 9.0
    assert "numbered 1 to 7" in llm.calls[1]["prompt"]
    assert len(llm.calls[1]["frames"]) == 7


def test_two_unclear_inspections_end_inconclusive_without_another_retry(
    synthetic_frames,
):
    llm = FakeLLM(
        [
            {"clear_view": False, "claims": []},
            {"clear_view": False, "claims": []},
            hidden_response(),
        ]
    )

    case = run(visit() + [event("person_exited", 12.0)], llm)

    assert case["severity"] == "inconclusive"
    assert len(llm.calls) == case["llm_calls"] == 2
    assert len(synthetic_frames) == 2


def test_valid_claim_is_kept_but_missing_unsent_and_hedged_claims_are_dropped(
    synthetic_frames,
):
    llm = FakeLLM(
        [
            {
                "clear_view": True,
                "claims": [
                    {
                        "claim": "item_taken",
                        "frame": 1,
                        "what": "takes a box",
                    },
                    {
                        "claim": "item_hidden",
                        "what": "puts it into a pocket",
                    },
                    {
                        "claim": "item_hidden",
                        "frame": 6,
                        "what": "puts it into a pocket",
                    },
                    {
                        "claim": "item_hidden",
                        "frame": 2,
                        "what": "appears to put it into a pocket",
                    },
                ],
            }
        ]
    )

    case = run(visit(), llm)

    assert [entry["claim"] for entry in case["evidence"]] == ["item_taken"]
    assert [entry["reason"] for entry in case["dropped"]] == [
        "no valid frame cited",
        "no valid frame cited",
        "hedged, not seen",
    ]
    assert "item_hidden" not in case["summary"]
    assert case["severity"] == "normal"


def test_claims_with_valid_citation_are_not_semantically_verified(synthetic_frames):
    """A frame index is checked, but the cited image is not checked against the text."""
    llm = FakeLLM(
        [
            {
                "clear_view": True,
                "claims": [
                    {
                        "claim": "item_hidden",
                        "frame": 1,
                        "what": "puts the item into a pocket",
                    }
                ],
            }
        ]
    )

    case = run(visit() + [event("person_exited", 12.0)], llm)

    assert case["severity"] == "L2"
    assert case["evidence"][0]["claim"] == "item_hidden"


def test_llm_exception_is_recorded_and_not_treated_as_suspicious_evidence(
    synthetic_frames,
):
    llm = FakeLLM([RuntimeError("provider unavailable")])

    case = run(
        visit() + [event("person_exited", 12.0)],
        llm,
        max_calls=1,
    )

    assert case["severity"] == "inconclusive"
    assert case["evidence"] == []
    assert case["dropped"][0]["reason"] == "llm error"
    assert "provider unavailable" in case["dropped"][0]["what"]
    assert case["llm_calls"] == len(llm.calls) == 1


def test_malformed_llm_response_returns_inconclusive(synthetic_frames):
    llm = FakeLLM([["not", "a", "response object"]])

    case = run(visit(), llm)

    assert case["severity"] == "inconclusive"
    assert case["evidence"] == []


def test_multiple_reaches_for_one_track_make_one_case_with_evidence_from_both(
    synthetic_frames,
):
    moments = [
        event("hand_reach_shelf", 2.0, llm_call=True),
        event("hand_leaves_shelf", 3.0),
        event("hand_reach_shelf", 7.0, llm_call=True),
        event("hand_leaves_shelf", 8.0),
        event("person_exited", 12.0),
    ]
    llm = FakeLLM(
        [
            {
                "clear_view": True,
                "claims": [
                    {"claim": "item_taken", "frame": 1, "what": "takes a box"}
                ],
            },
            hidden_response(),
        ]
    )

    case = run(moments, llm)

    assert case["track_id"] == 1
    assert case["severity"] == "L2"
    assert len(llm.calls) == case["llm_calls"] == 2
    assert [e["claim"] for e in case["evidence"]] == [
        "item_taken",
        "item_hidden",
    ]
    assert len(synthetic_frames) == 2


def test_two_interleaved_tracks_get_separate_evidence_and_cases(synthetic_frames):
    moments = [
        event("hand_reach_shelf", 1.0, 1, llm_call=True),
        event("hand_reach_shelf", 1.5, 2, llm_call=True),
        event("hand_leaves_shelf", 2.0, 1),
        event("hand_leaves_shelf", 2.5, 2),
        event("person_exited", 9.0, 1),
        event("person_exited", 10.0, 2),
    ]
    track_1 = run(
        moments,
        FakeLLM([hidden_response()]),
        track_id=1,
    )
    track_2 = run(
        moments,
        FakeLLM(
            [
                {
                    "clear_view": True,
                    "claims": [
                        {
                            "claim": "item_taken",
                            "frame": 2,
                            "what": "takes a bottle",
                        }
                    ],
                }
            ]
        ),
        track_id=2,
    )

    assert track_1["track_id"] == 1
    assert track_2["track_id"] == 2
    assert [e["claim"] for e in track_1["evidence"]] == ["item_hidden"]
    assert [e["claim"] for e in track_2["evidence"]] == ["item_taken"]
    assert track_1["severity"] == "L2"
    assert track_2["severity"] == "L1"


def test_other_tracks_counter_event_does_not_change_this_tracks_counter_check(
    synthetic_frames,
):
    moments = visit(track_id=1) + [event("at_counter", 4.0, track_id=2)]
    llm = FakeLLM([{"clear_view": True, "claims": []}])

    case = run(moments, llm, track_id=1)

    assert case["checks"]["counter_visit"] is None


def test_counter_before_reach_does_not_trigger_normal_visit_filter(
    synthetic_frames,
):
    llm = FakeLLM([hidden_response()])
    moments = [
        event("at_counter", 1.0),
        *visit(),
        event("person_exited", 12.0),
    ]

    case = run(moments, llm)

    assert len(llm.calls) == case["llm_calls"] == 1
    assert case["severity"] == "L1"
    assert case["checks"]["counter_visit"] is True


def test_exit_before_counter_does_not_trigger_normal_visit_filter(
    synthetic_frames,
):
    llm = FakeLLM([hidden_response()])
    moments = visit() + [
        event("person_exited", 5.0),
        event("at_counter", 8.0),
    ]

    case = run(moments, llm)

    assert len(llm.calls) == case["llm_calls"] == 1
    assert case["severity"] == "L2"
    assert case["checks"]["counter_visit"] is False


def test_unclear_reach_is_retried_even_if_another_reach_was_clear(
    synthetic_frames,
):
    moments = [
        event("hand_reach_shelf", 2.0, llm_call=True),
        event("hand_leaves_shelf", 3.0),
        event("hand_reach_shelf", 7.0, llm_call=True),
        event("hand_leaves_shelf", 8.0),
    ]
    llm = FakeLLM(
        [
            {"clear_view": True, "claims": []},
            {"clear_view": False, "claims": []},
            {"clear_view": True, "claims": []},
        ]
    )

    case = run(moments, llm)

    assert len(llm.calls) == case["llm_calls"] == 3
    assert [window["n"] for window in synthetic_frames] == [5, 5, 7]


def test_call_limit_includes_retry_attempts(synthetic_frames):
    llm = FakeLLM(
        [
            {"clear_view": False, "claims": []},
            {"clear_view": False, "claims": []},
        ]
    )

    case = run(visit(), llm, max_calls=1)

    assert len(llm.calls) == case["llm_calls"] == 1
    assert len(synthetic_frames) == 1
    assert case["severity"] == "inconclusive"


def test_multiple_reaches_never_exceed_configured_call_budget(synthetic_frames):
    moments = [
        event("hand_reach_shelf", t, llm_call=True)
        for t in (1.0, 5.0, 9.0, 13.0)
    ] + [event("person_exited", 20.0)]
    llm = FakeLLM([])

    case = run(moments, llm, max_calls=2)

    assert len(llm.calls) == case["llm_calls"] == 2
    assert len(synthetic_frames) == 2


def test_case_output_round_trips_through_sqlite_storage(
    synthetic_frames, tmp_path
):
    llm = FakeLLM(
        [
            {
                "clear_view": True,
                "claims": [
                    {"claim": "item_taken", "frame": 1, "what": "takes a box"},
                    {
                        "claim": "item_hidden",
                        "frame": 99,
                        "what": "puts it into a pocket",
                    },
                ],
            }
        ]
    )
    case = run(visit() + [event("person_exited", 12.0)], llm)
    case["video"] = "sample.mp4"
    conn = connect(tmp_path / "cases.db")

    case_id = save_case(conn, case)
    stored = list_cases(conn)[0]

    assert stored["id"] == case_id
    assert stored["video"] == "sample.mp4"
    assert stored["track_id"] == 1
    assert stored["severity"] == case["severity"] == "L1"
    assert stored["summary"] == case["summary"]
    assert stored["evidence"] == case["evidence"]
    assert stored["dropped"] == case["dropped"]
    assert stored["llm_calls"] == case["llm_calls"] == 1
    assert isinstance(stored["checks"], dict)
