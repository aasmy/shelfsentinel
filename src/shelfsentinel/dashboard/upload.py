"""dashboard section for analyzing an uploaded video: save it, find the zones, then run
scripts/analyze_video.py in the background and show how far it got."""

import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
try:  # the gemini key lives in .env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
UPLOADS = Path("data") / "uploads"
ZONE_CHOICES = {
    "yolo": "Find them (YOLO, free)",
    "gemini": "Find them (Gemini, 1 call)",
    "file": "Use a zones file",
}


def safe_name(name):
    """keep letters, numbers, dot, dash, underscore. no folders"""
    stem, suffix = os.path.splitext(Path(name).name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "video"
    return stem + (suffix.lower() if suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} else ".mp4")


def save_upload(uploaded):
    path = UPLOADS / safe_name(uploaded.name)
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_bytes(uploaded.getbuffer())
    return path


def run_dir(video):
    return ROOT / "runs" / Path(video).stem


def analyze_command(video, zones, db_path, use_llm, zones_only=False):
    """zones is "yolo" / "gemini" (find them) or the path of a zones file"""
    cmd = [sys.executable, "scripts/analyze_video.py", "--video", str(video), "--db", str(db_path)]
    cmd += ["--auto-zones", zones] if zones in ("yolo", "gemini") else ["--zones", str(zones)]
    if zones_only:
        cmd.append("--zones-only")
    if not use_llm:
        cmd.append("--no-llm")
    return cmd


def progress_lines(log_text):
    """the lines worth showing: the steps, each case and errors"""
    keep = []
    for line in log_text.splitlines():
        line = line.strip()
        if re.match(r"^(\d\)|case \d|zones |done|found \d+|\d+ events|step failed|no shelf|could not)", line) \
                or "Error" in line:
            keep.append(line)
    return keep


def case_line(line):
    """turns a run_agent line like: case 1 (db 5)  track 4  L1  calls=1  summary  into  Case #5 · L1 · summary"""
    m = re.match(r"case \d+ \(db (\d+)\)\s+track \S+\s+(\S+)\s+calls=\d+\s+(.*)", line)
    return f"Case #{m[1]} · {m[2]} · {m[3]}" if m else line


@st.fragment(run_every="2s")
def show_job():
    job = st.session_state.get("upload_job")
    if not job:
        return
    log_text = Path(job["log"]).read_text(errors="replace") if Path(job["log"]).exists() else ""
    code = job["proc"].poll()
    lines = progress_lines(log_text)
    if code is None:
        with st.status(f"Analyzing {Path(job['video']).name}… (pose tracking takes a while on a CPU)",
                       expanded=True, state="running"):
            for line in lines[-8:]:
                st.text(line)
    elif code == 0:
        cases = [case_line(line) for line in lines if line.startswith("case ")]
        st.success(f"Done: {len(cases)} case(s) from {Path(job['video']).name}. They're in the list below.")
        for line in cases:
            st.caption(line)
        if not job.get("shown"):
            job["shown"] = True
            st.rerun(scope="app")
    else:
        st.error(f"Analysis stopped ({lines[-1] if lines else 'see the log'}).")
        with st.expander("Log"):
            st.code(log_text[-3000:] or "empty")


def upload_section(db_path):
    job = st.session_state.get("upload_job")
    running = bool(job) and job["proc"].poll() is None
    with st.expander("Analyze a video", expanded=running or "upload_video" in st.session_state):
        if running:
            show_job()
            return
        uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
        if uploaded is None:
            st.session_state.pop("upload_video", None)
            st.caption("Upload a store video. It is checked the same way as the live camera: zones, "
                       "pose + tracking, then the case agent.")
            show_job()
            return
        if st.session_state.get("upload_name") != uploaded.name:  # new file
            st.session_state["upload_video"] = str(save_upload(uploaded))
            st.session_state["upload_name"] = uploaded.name
            st.session_state.pop("upload_zones_ready", None)
        video = st.session_state["upload_video"]

        left, right = st.columns(2)
        method = left.radio("Zones (shelf, counter, exit)", list(ZONE_CHOICES), format_func=ZONE_CHOICES.get,
                            key="upload_zone_method")
        zones_file = None
        if method == "file":
            zones_file = left.text_input("Zones file", value="configs/zones.yaml")
        has_key = bool(os.getenv("GEMINI_API_KEY"))
        use_llm = right.toggle("Check reaches with Gemini", value=has_key, disabled=not has_key,
                               help="uses the free quota (about 1 call per suspicious reach)"
                               if has_key else "no GEMINI_API_KEY in .env")

        if right.button("Find zones", use_container_width=True):
            with st.spinner("Finding zones…"):
                result = subprocess.run(
                    analyze_command(video, zones_file or method, db_path, use_llm, zones_only=True),
                    cwd=ROOT, capture_output=True, text=True)
            if result.returncode == 0:
                st.session_state["upload_zones_ready"] = video
            else:
                st.error((result.stderr or result.stdout).strip().splitlines()[-1] if
                         (result.stderr or result.stdout).strip() else "could not find zones")

        zones_png = run_dir(video) / "zones.png"
        if st.session_state.get("upload_zones_ready") == video and zones_png.exists():
            st.image(str(zones_png), caption="Zones that will be used. Wrong? Pick another option and find again.")

        if st.button("Analyze", type="primary", use_container_width=True):
            ready = st.session_state.get("upload_zones_ready") == video
            zones = run_dir(video) / "zones.yaml" if ready else (zones_file or method)
            log = run_dir(video) / "analyze.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as out:
                proc = subprocess.Popen(analyze_command(video, zones, db_path, use_llm), cwd=ROOT, stdout=out,
                                        stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            st.session_state["upload_job"] = {"proc": proc, "log": str(log), "video": video}
            st.rerun()
