# shelfsentinel
Agentic incident investigator for small retail stores. YOLO detection plus a LangGraph case agent that turns suspicious visits into evidence-backed case summaries.

## Run the case workflow

From the repository root, run event detection and then the case agent:

```powershell
python scripts/draw_zones.py data/test_vid7.mp4 --output configs/zones.yaml
python scripts/run_pipeline.py --video data/test_vid7.mp4 --zones configs/zones.yaml --no-llm
python scripts/run_agent.py --video data/test_vid7.mp4 --no-llm
```

The zone-drawing step is needed once for each camera layout.
The pipeline writes events to `runs/test_vid7/cases.jsonl`. The agent reads those
events and writes case summaries to `runs/test_vid7/agent_cases.jsonl` and
`runs/cases.db`. Use `--db path\to\cases.db` to select another SQLite database;
the agent saves every generated case, including normal and inconclusive cases.
Omit `--no-llm` to allow the agent to call Gemini (configure `GEMINI_API_KEY`
in `.env` first).

For investigated cases, accepted `item_taken` evidence can produce shelf clips;
completed checkout visits and exit events produce their own clips. The clips are
stored under `runs/case_<database-id>/`, while relative paths and event times
are saved in the SQLite `video_clips` column. Existing databases gain that
column automatically when opened. The dashboard plays available clips and
deleting a case also removes only its generated `runs/case_<id>/` directory;
the original source video is retained.

## Launch the case dashboard

Install the project dependencies, then run Streamlit from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/shelfsentinel/dashboard/app.py
```

The dashboard defaults to `runs/cases.db`. Enter a different path in the
sidebar to inspect another database.

## Evaluate

Measures what the proposal promises: thefts caught (recall), normal visits wrongly flagged (false alarm rate) and LLM calls per visit, for the agent and for a baseline that sends 5 fixed frames of the whole video to one LLM call. One labelled video = one visit.

1. Label the videos in a csv like `configs/labels_example.csv`:

   ```
   video,scenario,actor,theft,zones,notes
   data/visit_001.mp4,normal,A,no,configs/zones.yaml,picks item and pays at the counter
   data/visit_004.mp4,concealment,A,yes,configs/zones.yaml,item into left pocket and leaves without paying
   ```

   `theft` is yes/no. `actor` is a letter per person (no names), used for the bias audit.

2. Run it:

   ```
   python scripts/evaluate.py --labels data/labels.csv --run --baseline
   ```

   `--run` runs the pipeline and agent for videos that weren't run yet, `--baseline` adds the one-call baseline (1 LLM call per video). Results are cached per video so running it again doesn't spend quota (`--rerun` redoes them). Use `--no-llm` for a dry run.

3. Results: tables in the terminal (overall, by actor, by scenario) and `runs/eval/per_video.csv` + `runs/eval/summary.json` for the report.

## Live mode (webcam)

```
python scripts/run_live.py
```

Opens the webcam with detection, zones and events on screen. A visit starts when someone walks in and ends when nobody has been in view for 3 seconds (or after `person_exited`). The agent then checks that visit in the background, so the camera keeps running, and the case shows up on the dashboard (run `streamlit run src/shelfsentinel/dashboard/app.py` in another terminal). Press `q` to stop; the last visit still gets checked.

- Draw the zones for the camera first: record a few seconds, then `python scripts/draw_zones.py <that video>`. Don't move the camera after.
- `--camera 1` for another camera, `--no-llm` for no Gemini calls, `--max-calls` per visit (default 3).
- Every visit is saved in `runs/live/<visit>/` (video + events), so live sessions can be labelled and used in the evaluation too.
- `--source data/x.mp4 --no-show` plays a file as if it was the camera (for testing).
