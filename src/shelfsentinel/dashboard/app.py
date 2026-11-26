"""Streamlit page for reviewing cases one at a time: video, what happened, then Confirm or False alarm.

run from the repo root:
    streamlit run src/shelfsentinel/dashboard/app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shelfsentinel.dashboard.data import (
    VideoCleanupError,
    delete_case,
    load_cases,
    update_review,
)
from shelfsentinel.storage.db import DEFAULT_DB

SEVERITY_ORDER = {"L2": 0, "L1": 1, "inconclusive": 2, "normal": 3}
SEVERITY_NAMES = {"L2": "L2", "L1": "L1", "inconclusive": "Inconclusive", "normal": "Normal"}
REVIEW_NAMES = {"pending": "waiting for review", "confirmed": "confirmed", "false_alarm": "false alarm"}
CLAIM_NAMES = {"item_taken": "took an item", "item_hidden": "hid an item", "item_put_back": "put an item back"}
CLIP_TITLES = {
    "full_visit": "Whole visit",
    "shelf_interaction": "Shelf reach",
    "checkout": "Checkout",
    "exit": "Exit",
}


@st.fragment(run_every="4s")
def watch_for_new_cases(db_path, shown):
    """runs by itself every few seconds, reloads the page when a new case shows up (live mode)"""
    try:
        current = len(load_cases(db_path)[0])
    except Exception:
        return
    if current != shown:
        st.session_state["new_cases"] = current - shown
        st.rerun(scope="app")


def show_clip(raw_path):
    """only play clips that live inside the project folder"""
    if not isinstance(raw_path, str) or not raw_path:
        st.warning("Video evidence metadata has no valid file path.")
        return
    clip_path = Path(raw_path)
    if clip_path.is_absolute():
        st.warning(f"Video path must be relative: {clip_path}")
        return
    resolved_path = (Path.cwd() / clip_path).resolve()
    try:
        resolved_path.relative_to(Path.cwd().resolve())
    except ValueError:
        st.warning("Video path is outside the project directory.")
        return
    if not resolved_path.is_file():
        st.warning(f"Video evidence file is missing: {clip_path}")
        return
    # tall (portrait) videos go in the middle third so the buttons stay on screen
    try:
        import cv2

        capture = cv2.VideoCapture(str(resolved_path))
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        capture.release()
    except Exception:
        width = height = 0
    if height > width > 0:
        _, middle, _ = st.columns([1, 1.2, 1])
        middle.video(str(resolved_path))
    else:
        st.video(str(resolved_path))


def show_videos(case):
    clips = [clip for clip in case.get("video_clips") or [] if isinstance(clip, dict)]
    if not clips:
        st.write("No video clips are available for this case.")
        return
    titles = []
    for clip in clips:
        event_type = clip.get("event_type") if isinstance(clip.get("event_type"), str) else "video"
        title = CLIP_TITLES.get(event_type, event_type.replace("_", " ").title())
        titles.append(title if title not in titles else f"{title} {titles.count(title) + 1}")
    for tab, clip in zip(st.tabs(titles), clips):
        with tab:
            show_clip(clip.get("path"))
            start, end = clip.get("start_time"), clip.get("end_time")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (start, end)):
                st.caption(f"{max(start, 0):.1f}s – {end:.1f}s in the original video")


def decide(case, db_path, review, move_on):
    try:
        updated = update_review(db_path, case["id"], review)
    except Exception as exc:
        st.error(f"Could not save: {exc}")
        return
    if not updated:
        st.error("This case no longer exists.")
        return
    st.session_state["toast"] = f"Case #{case['id']} saved as {REVIEW_NAMES[review]}"
    if move_on:
        st.session_state["position"] = st.session_state.get("position", 0) + 1
    st.session_state.pop("selected_case_id", None)
    st.rerun()


def delete_section(case, db_path):
    pending_key = "pending_delete_case_id"
    if st.session_state.get(pending_key) != case["id"]:
        if st.button("Delete this case", key=f"delete-case-{case['id']}"):
            st.session_state[pending_key] = case["id"]
            st.rerun()
        return
    st.warning("Delete this case and its clips? The original video is not deleted.")
    confirm, cancel = st.columns(2)
    if confirm.button("Yes, delete", key=f"confirm-delete-{case['id']}", type="primary"):
        st.session_state.pop(pending_key, None)
        try:
            delete_case(db_path, case["id"])
        except VideoCleanupError as exc:
            st.session_state["case_delete_cleanup_error"] = str(exc)
        except Exception as exc:
            st.error(f"Could not delete case #{case['id']}: {exc}")
            return
        st.session_state["toast"] = f"Case #{case['id']} deleted"
        st.rerun()
    if cancel.button("Cancel", key=f"cancel-delete-{case['id']}"):
        st.session_state.pop(pending_key, None)
        st.rerun()


def show_case(case, db_path, waiting_view):
    st.subheader(f"Case #{case['id']}")
    show_videos(case)

    # what happened
    severity = SEVERITY_NAMES.get(case["severity"], case["severity"])
    finding = str(case["summary"]).split(". ")[0]
    st.markdown(f"#### {severity} – {finding}")
    details = st.columns(3)
    details[0].markdown(f"**Date/time:** {str(case['created_at'])[:16].replace('T', ' ')}")
    details[1].markdown(f"**Video:** {Path(str(case['video'])).name}")
    details[2].markdown(f"**Track ID:** {case['track_id']}")
    details[0].markdown(f"**Severity:** {severity}")
    details[1].markdown(f"**LLM calls:** {case['llm_calls']}")
    details[2].markdown(f"**Review:** {REVIEW_NAMES.get(case.get('review'), case.get('review'))}")
    evidence = sorted(case.get("evidence") or [], key=lambda e: e.get("frame_time") or 0)
    for item in evidence:
        t = item.get("frame_time")
        when = f"{t:.1f}s" if isinstance(t, (int, float)) else "?"
        st.markdown(f"- **{when}** {CLAIM_NAMES.get(item.get('claim'), item.get('claim'))}: {item.get('what', '')}")
    if not evidence and str(case["summary"]).strip() != finding.strip():
        st.write(case["summary"])

    # decision
    current = case.get("review", "pending")
    left, right, later, _ = st.columns([1, 1, 1, 1])
    if left.button("Confirm", key=f"review-confirmed-{case['id']}", type="primary", use_container_width=True,
                   disabled=current == "confirmed"):
        decide(case, db_path, "confirmed", move_on=not waiting_view)
    if right.button("False alarm", key=f"review-false_alarm-{case['id']}", use_container_width=True,
                    disabled=current == "false_alarm"):
        decide(case, db_path, "false_alarm", move_on=not waiting_view)
    if later.button("Decide later", key=f"review-pending-{case['id']}", use_container_width=True,
                    disabled=current == "pending"):
        decide(case, db_path, "pending", move_on=False)

    with st.expander("More details"):
        checks = case.get("checks") or {}
        yes_no = {True: "yes", False: "no"}
        st.write(
            f"Hands visible: {yes_no.get(checks.get('clear_view'), 'unknown')} · "
            f"counter visited: {yes_no.get(checks.get('counter_visit'), 'unknown')} · "
            f"left the store: {yes_no.get(checks.get('exited'), 'unknown')} · LLM calls: {case['llm_calls']}"
        )
        st.write(f"Full summary: {case['summary']}")
        dropped = case.get("dropped") or []
        st.write(f"Claims the guardrail removed ({len(dropped)}):")
        for item in dropped:
            st.markdown(f"- {item.get('claim') or 'response'} · _{item.get('reason', '')}_: "
                        f"{str(item.get('what', ''))[:200]}")
        st.caption(f"Source video: {case['video']}")
        delete_section(case, db_path)


def main():
    st.set_page_config(page_title="ShelfSentinel", layout="centered", initial_sidebar_state="collapsed")
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"))
    if st.session_state.pop("new_cases", 0):
        st.toast("New case from the camera")
    cleanup_error = st.session_state.pop("case_delete_cleanup_error", None)
    if cleanup_error:
        st.error(cleanup_error)

    with st.sidebar:
        db_path = st.text_input(
            "SQLite database path",
            value=os.environ.get("SHELF_SENTINEL_DB", DEFAULT_DB),
            help="Defaults to runs/cases.db. Relative paths use the current directory.",
        )
        auto_refresh = st.toggle("Live updates", value=True, help="Check for new cases every few seconds")

    try:
        cases, _ = load_cases(db_path)
    except Exception as exc:
        st.error(f"Could not load cases from {db_path}: {exc}")
        st.stop()
    if auto_refresh:
        watch_for_new_cases(db_path, len(cases))

    waiting = [c for c in cases if c.get("review") == "pending"]
    title, choice = st.columns([3, 2])
    title.markdown("### ShelfSentinel")
    view = choice.segmented_control("Show", ["Waiting", "All", "Reviewed"], default="Waiting",
                                    key="view", label_visibility="collapsed") or "Waiting"
    # L2 = possible concealment and left without the counter, these show up as a notification (proposal)
    pending_l2 = [c for c in waiting if c["severity"] == "L2"]
    if pending_l2:
        st.error(f"{len(pending_l2)} L2 case(s) need review: "
                 + ", ".join(f"#{c['id']} (track {c['track_id']})" for c in pending_l2))

    shown = {
        "Waiting": waiting,
        "All": cases,
        "Reviewed": [c for c in cases if c.get("review") != "pending"],
    }[view]
    counts = {sev: sum(c["severity"] == sev for c in shown) for sev in SEVERITY_ORDER}
    severity = st.segmented_control(
        "Severity", ["All", *SEVERITY_ORDER], default="All", key="severity",
        format_func=lambda sev: "All" if sev == "All" else f"{SEVERITY_NAMES[sev]} ({counts[sev]})",
        label_visibility="collapsed") or "All"
    if severity != "All":
        shown = [c for c in shown if c["severity"] == severity]
    shown = sorted(shown, key=lambda c: (SEVERITY_ORDER.get(c["severity"], 4), -c["id"]))

    if not cases:
        st.info("No cases yet. They show up here when the camera or the agent finds something.")
        return
    if not shown:
        if view == "Waiting":
            st.success("All cases are reviewed.")
        else:
            st.info("No cases here.")
        return

    # which case: a picked one (e.g. from a test), otherwise the current position
    ids = [c["id"] for c in shown]
    if st.session_state.get("selected_case_id") in ids:
        st.session_state["position"] = ids.index(st.session_state["selected_case_id"])
    position = min(max(st.session_state.get("position", 0), 0), len(shown) - 1)
    st.session_state["position"] = position
    case = shown[position]

    previous, label, following = st.columns([1, 2, 1])
    if previous.button("< Previous", use_container_width=True, disabled=position == 0):
        st.session_state["position"] = position - 1
        st.session_state.pop("selected_case_id", None)
        st.rerun()
    label.markdown(
        f"<div style='text-align:center;padding-top:6px'><b>Case #{case['id']}</b> · {position + 1} of {len(shown)}"
        f" · {len(waiting)} waiting</div>",
        unsafe_allow_html=True,
    )
    if following.button("Next >", use_container_width=True, disabled=position >= len(shown) - 1):
        st.session_state["position"] = position + 1
        st.session_state.pop("selected_case_id", None)
        st.rerun()

    show_case(case, db_path, waiting_view=view == "Waiting")


if __name__ == "__main__":
    main()
