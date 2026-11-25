"""Database access helpers used by the dashboard."""

from contextlib import closing

from shelfsentinel.storage.db import (
    connect,
    delete_case as delete_stored_case,
    list_cases,
    set_review,
)


def load_cases(db_path):
    """Load cases and severity totals using one short-lived connection."""
    with closing(connect(db_path)) as conn:
        cases = list_cases(conn)
    counts = {}
    for case in cases:
        counts[case["severity"]] = counts.get(case["severity"], 0) + 1
    return cases, counts


def update_review(db_path, case_id, review):
    """Persist a review change and close the connection."""
    with closing(connect(db_path)) as conn:
        with conn:
            return set_review(conn, case_id, review, commit=False)


def delete_case(db_path, case_id):
    """Delete a case and close the database connection."""
    with closing(connect(db_path)) as conn:
        with conn:
            return delete_stored_case(conn, case_id, commit=False)
