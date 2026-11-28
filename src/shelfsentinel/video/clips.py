"""Extract short, event-centred clips for investigation cases."""

import logging
import math
import os
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _event_time(event):
    value = event.get("t", event.get("timestamp"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def extract_video_clip(video_path, start_time, end_time, output_path):
    """Write a source-video interval to MP4 and return its actual frame bounds."""
    info = _write_clip(video_path, start_time, end_time, output_path)
    info["browser_playable"] = _make_browser_playable(Path(output_path))
    return info


def _make_browser_playable(path):
    """Re-encode the clip as H.264.

    opencv writes mp4v, which browsers (and discord/streamlit) cant play, they only show one frame.
    imageio-ffmpeg ships its own ffmpeg so this works on windows too. if it fails we keep the mp4v file,
    it still plays after downloading.
    """
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("ffmpeg not available, clip stays mp4v: %s", exc)
        return False

    tmp = path.with_name(path.stem + ".h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning("could not re-encode %s to h264: %s", path, exc)
        tmp.unlink(missing_ok=True)
        return False


def _write_clip(video_path, start_time, end_time, output_path):
    for value in (start_time, end_time):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("clip timestamps must be finite numbers")
        if not math.isfinite(float(value)):
            raise ValueError("clip timestamps must be finite numbers")
    if end_time <= start_time:
        raise ValueError("clip end time must be after its start time")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"source video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"could not open source video: {video_path}")

    output_path = Path(output_path)
    writer = None
    frames_written = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            raise OSError(f"source video has invalid metadata: {video_path}")

        duration = total_frames / fps
        actual_start = min(max(float(start_time), 0.0), duration)
        actual_end = min(max(float(end_time), 0.0), duration)
        if actual_end <= actual_start:
            raise ValueError("clip interval does not overlap the source video")

        first_frame = min(int(math.floor(actual_start * fps)), total_frames - 1)
        end_frame = min(int(math.ceil(actual_end * fps)), total_frames)
        end_frame = max(end_frame, first_frame + 1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for codec in ("mp4v", "avc1"):
            candidate = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
            output_path.unlink(missing_ok=True)
        if writer is None:
            raise OSError(f"could not initialize an MP4 writer for {output_path}")

        seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if (
            not seek_ok
            or not math.isfinite(current_frame)
            or abs(current_frame - first_frame) > 1
        ):
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise OSError(f"could not reopen source video: {video_path}")
            for _ in range(first_frame):
                ok, _ = cap.read()
                if not ok:
                    raise OSError(f"could not seek to clip start in {video_path}")
        for _ in range(first_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1

        if frames_written == 0:
            raise OSError(f"no frames could be extracted from {video_path}")
        return {
            "actual_start_time": first_frame / fps,
            "actual_end_time": min(end_frame, first_frame + frames_written) / fps,
            "frames": frames_written,
            "fps": fps,
            "width": width,
            "height": height,
        }
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def _shelf_interactions(events, track_id):
    track_events = [
        event
        for event in events
        if event.get("track_id") == track_id
        and event.get("event") in {"hand_reach_shelf", "hand_leaves_shelf"}
    ]
    track_events.sort(key=lambda event: _event_time(event) or 0.0)
    interactions = []

    for index, event in enumerate(track_events):
        if event.get("event") != "hand_reach_shelf" or not event.get("llm_call"):
            continue
        reach_time = _event_time(event)
        if reach_time is None:
            continue
        # same as the agent's frame window: the wrist flickers on the zone edge, so use the last
        # leave before the next real reach (within 10s), not the first one
        leave_event = None
        for following in track_events[index + 1 :]:
            following_time = _event_time(following)
            if following_time is not None and following_time - reach_time > 10.0:
                break
            if following.get("event") == "hand_reach_shelf" and following.get("llm_call"):
                break
            if following.get("event") == "hand_leaves_shelf":
                leave_event = following
        leave_time = _event_time(leave_event) if leave_event else None
        if leave_time is not None and leave_time >= reach_time:
            interactions.append((reach_time, leave_time))
    return interactions


def _checkout_visits(events, track_id):
    track_events = [
        event
        for event in events
        if event.get("track_id") == track_id
        and event.get("event") in {"at_counter", "leave_counter"}
    ]
    track_events.sort(key=lambda event: _event_time(event) or 0.0)
    visits = []

    for index, event in enumerate(track_events):
        if event.get("event") != "at_counter":
            continue
        start = _event_time(event)
        if start is None:
            continue
        for following in track_events[index + 1 :]:
            if following.get("event") == "at_counter":
                break
            if following.get("event") == "leave_counter":
                end = _event_time(following)
                if end is not None and end >= start:
                    visits.append((start, end))
                break
    return visits


def create_case_clips(
    video_path,
    case_id,
    track_id,
    events,
    evidence,
    runs_dir="runs",
    full_visit=False,
    max_unchecked=3,
):
    """Create clips justified by accepted evidence and matching track events.

    Clip errors are logged and skipped so they do not prevent case persistence.
    Metadata start/end times retain the requested event window; actual clipped
    frame bounds are recorded separately after source-boundary clamping.
    """
    clips = []
    output_dir = Path(runs_dir) / f"case_{case_id}"
    used_interactions = set()
    interaction_pairs = _shelf_interactions(events, track_id)
    # any shelf claim gets a clip, item_hidden is the one that matters most
    taken_evidence = [
        claim
        for claim in evidence
        if claim.get("claim") in {"item_taken", "item_hidden", "item_put_back", "item_taken_out"}
    ]

    for claim in taken_evidence:
        frame_time = _event_time({"t": claim.get("frame_time")})
        if frame_time is None:
            logger.warning(
                "Skipping item_taken evidence without a valid frame time "
                "for track %s in case %s",
                track_id,
                case_id,
            )
            continue
        matching = []
        for index, (reach_time, leave_time) in enumerate(interaction_pairs):
            if index in used_interactions:
                continue
            frame_start = max(0.0, reach_time - 0.5)
            frame_end = min(leave_time + 3.0, frame_start + 10.0)
            if frame_start <= frame_time <= frame_end:
                matching.append((index, reach_time, leave_time))
        if not matching:
            logger.warning(
                "No matching shelf reach/leave pair for item_taken evidence "
                "on track %s in case %s",
                track_id,
                case_id,
            )
            continue
        index, reach_time, leave_time = min(
            matching,
            key=lambda item: (
                abs(frame_time - item[1]),
                item[1],
            ),
        )
        used_interactions.add(index)
        start_time, end_time = reach_time - 2.0, leave_time + 3.0
        output_path = output_dir / f"shelf_interaction_{index + 1}.mp4"
        clips.append(
            _create_clip_metadata(
                video_path,
                track_id,
                "shelf_interaction",
                start_time,
                end_time,
                output_path,
                {"hand_reach_shelf": reach_time, "hand_leaves_shelf": leave_time},
            )
        )

    # no evidence (llm not used or out of calls): still show the suspicious reaches so the owner can look.
    # reaches followed by hand_to_body first
    if not used_interactions and interaction_pairs:
        body_times = sorted(
            t for t in (
                _event_time(e) for e in events
                if e.get("track_id") == track_id and e.get("event") == "hand_to_body"
            ) if t is not None
        )

        # each hand_to_body belongs to the last reach before it
        flagged = set()
        for t in body_times:
            before = [i for i, (reach_time, _) in enumerate(interaction_pairs) if reach_time <= t]
            if before:
                flagged.add(max(before, key=lambda i: interaction_pairs[i][0]))

        order = sorted(range(len(interaction_pairs)), key=lambda i: (i not in flagged, i))
        for index in sorted(order[:max_unchecked]):
            reach_time, leave_time = interaction_pairs[index]
            clips.append(
                _create_clip_metadata(
                    video_path,
                    track_id,
                    "shelf_interaction",
                    reach_time - 2.0,
                    leave_time + 3.0,
                    output_dir / f"shelf_interaction_{index + 1}.mp4",
                    {"hand_reach_shelf": reach_time, "hand_leaves_shelf": leave_time},
                )
            )

    for index, (counter_time, leave_time) in enumerate(
        _checkout_visits(events, track_id), start=1
    ):
        clips.append(
            _create_clip_metadata(
                video_path,
                track_id,
                "checkout",
                counter_time - 1.0,
                leave_time + 1.0,
                output_dir / f"checkout_{index}.mp4",
                {"at_counter": counter_time, "leave_counter": leave_time},
            )
        )

    exit_events = [
        event
        for event in events
        if event.get("track_id") == track_id
        and event.get("event") == "person_exited"
    ]
    exit_events.sort(key=lambda event: _event_time(event) or 0.0)
    seen_exit_times = set()
    for index, event in enumerate(exit_events, start=1):
        exit_time = _event_time(event)
        if exit_time is None:
            logger.warning(
                "Skipping exit event with invalid timestamp in case %s",
                case_id,
            )
            continue
        if exit_time in seen_exit_times:
            logger.warning(
                "Skipping duplicate exit event at %.3fs in case %s",
                exit_time,
                case_id,
            )
            continue
        seen_exit_times.add(exit_time)
        clips.append(
            _create_clip_metadata(
                video_path,
                track_id,
                "exit",
                exit_time - 3.0,
                exit_time + 1.0,
                output_dir / f"exit_{index}.mp4",
                {"person_exited": exit_time},
            )
        )

    # live visits are short, so keep the whole visit as one video too
    if full_visit:
        duration = _video_duration(video_path)
        if duration and duration <= 900:
            clips.insert(0, _create_clip_metadata(
                video_path, track_id, "full_visit", 0.0, duration, output_dir / "full_visit.mp4", {},
            ))
    return [clip for clip in clips if clip is not None]


def _video_duration(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return frames / fps if fps else None
    finally:
        capture.release()


def _create_clip_metadata(
    video_path,
    track_id,
    event_type,
    start_time,
    end_time,
    output_path,
    event_times,
):
    try:
        extracted = extract_video_clip(
            video_path,
            start_time,
            end_time,
            output_path,
        )
        relative_path = os.path.relpath(output_path.resolve(), Path.cwd().resolve())
    except Exception as exc:
        logger.warning(
            "Could not create %s clip for track %s: %s",
            event_type,
            track_id,
            exc,
        )
        try:
            output_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "Could not remove partial clip %s: %s",
                output_path,
                cleanup_error,
            )
        return None

    return {
        "event_type": event_type,
        "track_id": track_id,
        "start_time": start_time,
        "end_time": end_time,
        "event_times": event_times,
        "actual_start_time": extracted["actual_start_time"],
        "actual_end_time": extracted["actual_end_time"],
        "path": relative_path,
    }
