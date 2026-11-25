"""Small Streamlit interface for browsing and reviewing stored cases."""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shelfsentinel.dashboard.data import load_cases, update_review
from shelfsentinel.storage.db import DEFAULT_DB, REVIEW_STATUSES


def main():
    st.set_page_config(page_title="ShelfSentinel Cases", layout="wide")
    st.title("ShelfSentinel Case Review")
    st.caption("Browse investigation cases and record a human review status.")

    with st.sidebar:
        st.header("Database")
        db_path = st.text_input(
            "SQLite database path",
            value=os.environ.get("SHELF_SENTINEL_DB", DEFAULT_DB),
            help="Defaults to runs/cases.db. Relative paths use the current directory.",
        )

    try:
        cases, counts = load_cases(db_path)
    except Exception as exc:
        st.error(f"Could not load cases from {db_path}: {exc}")
        st.stop()

    st.subheader("Overview")
    metrics = st.columns(5)
    metrics[0].metric("Total cases", len(cases))
    for column, severity in zip(metrics[1:], ("L1", "L2", "inconclusive", "normal")):
        column.metric(severity, counts.get(severity, 0))

    if not cases:
        st.info("No cases are stored in this database yet.")
        return

    severity_options = ["All"] + sorted(counts)
    severity_filter = st.selectbox("Filter by severity", severity_options)
    filtered_cases = [
        case
        for case in cases
        if severity_filter == "All" or case["severity"] == severity_filter
    ]

    if not filtered_cases:
        st.info("No cases match this severity.")
        return

    labels = {
        case["id"]: (
            f"#{case['id']} · {case['severity']} · "
            f"{case['video']} · track {case['track_id']}"
        )
        for case in filtered_cases
    }
    selected_id = st.selectbox(
        "Select a case",
        options=list(labels),
        format_func=labels.__getitem__,
    )
    case = next(item for item in filtered_cases if item["id"] == selected_id)

    st.subheader(f"Case #{case['id']}")
    detail_columns = st.columns(3)
    detail_columns[0].write(f"**Date/time:** {case['created_at']}")
    detail_columns[1].write(f"**Video:** {case['video']}")
    detail_columns[2].write(f"**Track ID:** {case['track_id']}")
    detail_columns[0].write(f"**Severity:** {case['severity']}")
    detail_columns[1].write(f"**LLM calls:** {case['llm_calls']}")
    detail_columns[2].write(f"**Review:** {case['review']}")

    st.markdown("**Summary**")
    st.write(case["summary"])

    checks = case["checks"]
    st.markdown("**Counter and exit checks**")
    check_columns = st.columns(2)
    check_columns[0].write(f"Counter visited: `{checks.get('counter_visit', 'unknown')}`")
    check_columns[1].write(f"Exited: `{checks.get('exited', 'unknown')}`")
    with st.expander("All checks"):
        st.json(checks)

    st.markdown("**Accepted evidence**")
    if case["evidence"]:
        for index, evidence in enumerate(case["evidence"], start=1):
            frame_time = evidence.get("frame_time")
            label = f"{index}. {evidence.get('claim', 'claim')}"
            if frame_time is not None:
                label += f" · frame time {frame_time}s"
            with st.expander(label):
                st.write(evidence.get("what", ""))
                st.json(evidence)
    else:
        st.write("No accepted evidence.")

    st.markdown("**Dropped claims**")
    if case["dropped"]:
        for index, dropped in enumerate(case["dropped"], start=1):
            label = (
                f"{index}. {dropped.get('claim') or 'response'} · "
                f"{dropped.get('reason', 'no reason recorded')}"
            )
            with st.expander(label):
                st.write(dropped.get("what", ""))
                st.json(dropped)
    else:
        st.write("No dropped claims.")

    st.markdown("**Update human review**")
    review_options = sorted(REVIEW_STATUSES)
    current_review = case.get("review", "pending")
    review_index = review_options.index(current_review)
    new_review = st.selectbox(
        "Review status",
        review_options,
        index=review_index,
        key=f"review-{case['id']}",
    )
    if st.button("Save review status"):
        try:
            updated = update_review(db_path, case["id"], new_review)
        except Exception as exc:
            st.error(f"Could not update review status: {exc}")
        else:
            if updated:
                st.success("Review status saved.")
                st.rerun()
            else:
                st.error("Case no longer exists in the selected database.")


if __name__ == "__main__":
    main()
