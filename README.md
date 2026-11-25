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

## Launch the case dashboard

Install the project dependencies, then run Streamlit from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run src/shelfsentinel/dashboard/app.py
```

The dashboard defaults to `runs/cases.db`. Enter a different path in the
sidebar to inspect another database.
