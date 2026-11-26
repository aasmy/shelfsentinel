"""Small Streamlit interface for browsing and reviewing stored cases."""

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


REVIEW_LABELS = {"confirmed": "Confirm", "false_alarm": "False alarm", "pending": "Pending"}


def review_buttons(case, db_path):
    """three plain buttons instead of a dropdown + save"""
    current = case.get("review", "pending")
    st.markdown(f"**Review:** {REVIEW_LABELS.get(current, current)}")
    columns = st.columns([1, 1, 1, 3])
    for column, value in zip(columns, ("confirmed", "false_alarm", "pending")):
        if column.button(REVIEW_LABELS[value], key=f"review-{value}-{case['id']}", disabled=current == value,
                         use_container_width=True):
            try:
                updated = update_review(db_path, case["id"], value)
            except Exception as exc:
                st.error(f"Could not update review status: {exc}")
            else:
                if updated:
                    st.toast(f"Case #{case['id']}: {REVIEW_LABELS[value].lower()}")
                    st.rerun()
                else:
                    st.error("Case no longer exists in the selected database.")


def main():
    st.set_page_config(page_title="ShelfSentinel Cases", layout="wide")
    st.title("ShelfSentinel Case Review")
    st.caption("Browse investigation cases and record a human review status.")
    if st.session_state.pop("case_delete_success", False):
        st.success("Case deleted.")
    cleanup_error = st.session_state.pop("case_delete_cleanup_error", None)
    if cleanup_error:
        st.error(cleanup_error)

    with st.sidebar:
        st.header("Database")
        db_path = st.text_input(
            "SQLite database path",
            value=os.environ.get("SHELF_SENTINEL_DB", DEFAULT_DB),
            help="Defaults to runs/cases.db. Relative paths use the current directory.",
        )
        auto_refresh = st.toggle("Live updates", value=True, help="Check for new cases every few seconds")

    try:
        cases, counts = load_cases(db_path)
    except Exception as exc:
        st.error(f"Could not load cases from {db_path}: {exc}")
        st.stop()

    if auto_refresh:
        watch_for_new_cases(db_path, len(cases))
    if st.session_state.pop("new_cases", 0):
        st.toast("New case from the camera")

    # L2 = possible concealment and left without the counter, these show up as a notification (proposal)
    pending_l2 = [c for c in cases if c["severity"] == "L2" and c.get("review") == "pending"]
    if pending_l2:
        st.error(
            f"{len(pending_l2)} L2 case(s) need review: "
            + ", ".join(f"#{c['id']} (track {c['track_id']})" for c in pending_l2)
        )

    st.subheader("Overview")
    metrics = st.columns(5)
    metrics[0].metric("Total cases", len(cases))
    for column, severity in zip(metrics[1:], ("L1", "L2", "inconclusive", "normal")):
        column.metric(severity, counts.get(severity, 0))

    if not cases:
        st.info("No cases are stored in this database yet.")
        return

    filter_columns = st.columns(2)
    severity_filter = filter_columns[0].segmented_control(
        "Severity", ["All", "L2", "L1", "inconclusive", "normal"], default="All", key="severity_filter",
    ) or "All"
    review_filter = filter_columns[1].segmented_control(
        "Review", ["All", "pending", "confirmed", "false_alarm"], default="All", key="review_filter",
    ) or "All"
    filtered_cases = [
        case
        for case in cases
        if (severity_filter == "All" or case["severity"] == severity_filter)
        and (review_filter == "All" or case.get("review") == review_filter)
    ]
    # most serious first, then the ones nobody reviewed yet
    rank = {"L2": 0, "L1": 1, "inconclusive": 2, "normal": 3}
    filtered_cases.sort(key=lambda c: (rank.get(c["severity"], 4), c.get("review") != "pending", -c["id"]))

    if not filtered_cases:
        st.info("No cases match these filters.")
        return

    # click a row to open the case
    table = st.dataframe(
        [
            {
                "ID": case["id"],
                "Severity": case["severity"],
                "Review": case.get("review", "pending"),
                "Track": case["track_id"],
                "Video": Path(str(case["video"])).name,
                "Time": str(case["created_at"])[:16].replace("T", " "),
                "Summary": case["summary"],
            }
            for case in filtered_cases
        ],
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        height=min(38 + 35 * len(filtered_cases), 300),
        key="case_table",
    )
    selected_rows = table.selection.rows if table is not None else []
    if selected_rows:
        st.session_state["selected_case_id"] = filtered_cases[selected_rows[0]]["id"]
    ids = [c["id"] for c in filtered_cases]
    selected_id = st.session_state.get("selected_case_id")
    if selected_id not in ids:
        selected_id = ids[0]
    case = next(item for item in filtered_cases if item["id"] == selected_id)
    st.caption("Click a row to open that case.")

    st.subheader(f"Case #{case['id']}")
    detail_columns = st.columns(3)
    detail_columns[0].write(f"**Date/time:** {case['created_at']}")
    detail_columns[1].write(f"**Video:** {Path(str(case['video'])).name}")
    detail_columns[2].write(f"**Track ID:** {case['track_id']}")
    detail_columns[0].write(f"**Severity:** {case['severity']}")
    detail_columns[1].write(f"**LLM calls:** {case['llm_calls']}")

    st.markdown("**Summary**")
    st.write(case["summary"])

    review_buttons(case, db_path)

    checks = case["checks"]
    st.markdown("**Counter and exit checks**")
    check_columns = st.columns(2)
    check_columns[0].write(f"Counter visited: `{checks.get('counter_visit', 'unknown')}`")
    check_columns[1].write(f"Exited: `{checks.get('exited', 'unknown')}`")
    with st.expander("All checks"):
        st.json(checks)

    st.markdown("**Accepted evidence**")
    if case["evidence"]:
        for evidence in sorted(case["evidence"], key=lambda e: e.get("frame_time") or 0):
            frame_time = evidence.get("frame_time")
            when = f"{frame_time}s" if frame_time is not None else "?"
            st.markdown(f"- **{when}** · {evidence.get('claim', 'claim')}: {evidence.get('what', '')}")
    else:
        st.write("No accepted evidence.")

    st.markdown("**Video Evidence**")
    video_clips = case.get("video_clips", [])
    if not video_clips:
        st.write("No video clips are available for this case.")
    else:
        clip_titles = {
            "full_visit": "Whole Visit",
            "shelf_interaction": "Shelf Interaction",
            "checkout": "Checkout",
            "exit": "Exit",
        }
        for clip in video_clips:
            if not isinstance(clip, dict):
                st.warning("Skipping malformed video clip metadata.")
                continue
            event_type = clip.get("event_type", "video")
            if not isinstance(event_type, str):
                event_type = "video"
            title = clip_titles.get(
                event_type,
                event_type.replace("_", " ").title(),
            )
            st.markdown(f"**{title}**")
            start_time = clip.get("start_time")
            end_time = clip.get("end_time")
            if (
                isinstance(start_time, (int, float))
                and not isinstance(start_time, bool)
                and isinstance(end_time, (int, float))
                and not isinstance(end_time, bool)
            ):
                st.caption(
                    f"Track {clip.get('track_id', case['track_id'])} · "
                    f"{start_time:.2f}s–{end_time:.2f}s"
                )
            raw_path = clip.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                st.warning("Video evidence metadata has no valid file path.")
                continue
            clip_path = Path(raw_path)
            if clip_path.is_absolute():
                st.warning(f"Video path must be relative: {clip_path}")
                continue
            resolved_path = (Path.cwd() / clip_path).resolve()
            try:
                resolved_path.relative_to(Path.cwd().resolve())
            except ValueError:
                st.warning("Video path is outside the project directory.")
                continue
            if not resolved_path.is_file():
                st.warning(f"Video evidence file is missing: {clip_path}")
                continue
            video_column, _ = st.columns([2, 3])
            video_column.video(str(resolved_path))

    with st.expander(f"Dropped claims ({len(case['dropped'])})"):
        if case["dropped"]:
            for dropped in case["dropped"]:
                st.markdown(f"- {dropped.get('claim') or 'response'} · _{dropped.get('reason', 'no reason recorded')}_: "
                            f"{str(dropped.get('what', ''))[:200]}")
        else:
            st.write("No dropped claims.")

    pending_delete_key = "pending_delete_case_id"
    if st.button("Delete Case", key=f"delete-case-{case['id']}"):
        st.session_state[pending_delete_key] = case["id"]
        st.rerun()

    if st.session_state.get(pending_delete_key) == case["id"]:
        st.warning(
            f"Delete case #{case['id']} for {case['video']} "
            f"(track {case['track_id']}, severity {case['severity']})? "
            "This permanently removes the case record and its generated "
            "evidence clips. The source video will not be deleted."
        )
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Confirm Delete",
            key=f"confirm-delete-{case['id']}",
            type="primary",
        ):
            try:
                deleted = delete_case(db_path, case["id"])
            except VideoCleanupError as exc:
                st.session_state.pop(pending_delete_key, None)
                st.session_state["case_delete_cleanup_error"] = str(exc)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete case #{case['id']}: {exc}")
            else:
                st.session_state.pop(pending_delete_key, None)
                if deleted:
                    st.session_state["case_delete_success"] = True
                else:
                    st.session_state["case_delete_missing"] = case["id"]
                st.rerun()
        if cancel_column.button("Cancel", key=f"cancel-delete-{case['id']}"):
            st.session_state.pop(pending_delete_key, None)
            st.rerun()

    missing_case_id = st.session_state.pop("case_delete_missing", None)
    if missing_case_id is not None:
        st.warning(
            f"Case #{missing_case_id} was already deleted or no longer exists."
        )


if __name__ == "__main__":
    main()
