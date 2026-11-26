"""Database access helpers used by the dashboard."""

from contextlib import closing
import json
import shutil
from pathlib import Path

from shelfsentinel.storage.db import (
    connect,
    delete_case as delete_stored_case,
    list_cases,
    set_review,
)


class VideoCleanupError(OSError):
    """The case row was deleted, but its evidence directory could not be removed."""


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


def delete_case(db_path, case_id, runs_dir="runs"):
    """Delete a case, then remove only its validated case evidence directory."""
    if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 1:
        raise ValueError("case_id must be a positive integer")

    runs_root = Path(runs_dir).resolve()
    case_dir = runs_root / f"case_{case_id}"
    if case_dir.resolve() != case_dir:
        raise ValueError("case evidence directory must not be a symbolic link")
    if case_dir.parent != runs_root or case_dir.name != f"case_{case_id}":
        raise ValueError("case evidence directory is outside the runs directory")

    with closing(connect(db_path)) as conn:
        with conn:
            row = conn.execute(
                "SELECT video, video_clips FROM cases WHERE id = ?",
                (case_id,),
            ).fetchone()
            if row is None:
                return False
            source_path = Path(row["video"])
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            source_path = source_path.resolve()
            clips = json.loads(row["video_clips"] or "[]")
            deleted = delete_stored_case(conn, case_id, commit=False)
    if not deleted:
        return False

    try:
        source_is_inside = False
        if case_dir.is_dir() and source_path.is_file():
            try:
                source_path.relative_to(case_dir)
                source_is_inside = True
            except ValueError:
                pass

        if source_is_inside:
            _remove_case_clips(case_dir, clips, source_path)
            try:
                case_dir.rmdir()
            except OSError as exc:
                raise VideoCleanupError(
                    f"Case #{case_id} was deleted, but its source video is "
                    f"inside the evidence directory; retained {source_path}"
                ) from exc
        elif case_dir.is_dir():
            shutil.rmtree(case_dir)
        elif case_dir.exists():
            raise OSError(f"case evidence path is not a directory: {case_dir}")
    except VideoCleanupError:
        raise
    except OSError as exc:
        raise VideoCleanupError(
            f"Case #{case_id} was deleted, but evidence cleanup failed: {exc}"
        ) from exc
    return True


def _remove_case_clips(case_dir, clips, protected_path):
    """Remove only stored clip files that resolve inside this case directory."""
    for clip in clips:
        path = Path(clip.get("path", ""))
        if path.is_absolute():
            continue
        candidate = (Path.cwd() / path).resolve()
        try:
            candidate.relative_to(case_dir)
        except ValueError:
            continue
        if candidate == protected_path:
            continue
        if candidate.is_file():
            candidate.unlink()
