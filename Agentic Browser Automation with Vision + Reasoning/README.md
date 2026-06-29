# Agentic Browser

An offline AI agent that controls a real browser by seeing the screen.
Given a high-level research goal, it takes screenshots, understands the UI,
decides what to click or type, and repeats until the task is complete.
Produces a structured report with findings, reasoning trace, and screenshots.

**Fully offline. No cloud APIs. No cost per run.**

---

## Architecture

```
                          User Goal
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │           FastAPI  (:8000)              │
        │         POST /tasks  (ndjson stream)    │
        └──────────────────┬──────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────┐
        │         LangGraph Agent Loop            │
        │                                         │
        │  OBSERVE → THINK → ACT → VERIFY ──┐    │
        │     ▲                              │    │
        │     └──────────────────────────────┘    │
        │              (repeat)                   │
        └──────┬──────────────┬───────────────────┘
               │              │
               ▼              ▼
          llava:13b      llama3.1:8b
          (vision)       (reasoning)
          screenshot     action plan
          analysis
               │              │
               └──────┬───────┘
                      ▼
                  Playwright
                  (Chromium)
                      │
                      ▼
              SQLite + Reports
```

---

## Tech Stack

- Python 3.11+
- Ollama (local inference — no cloud)
  - llava:13b — vision / screenshot analysis
  - llama3.1:8b-instruct-q4_0 — reasoning / action planning
- LangGraph 0.1.19 — agent loop orchestration
- Playwright — browser automation (Chromium)
- FastAPI — REST API with ndjson streaming
- Streamlit — live dashboard
- SQLite — task history and report storage

---

## Hardware Requirements

Minimum: RTX 2060 (6GB VRAM), 16GB RAM
Models run sequentially — vision first, then reasoning. Not simultaneously.
llava:13b is approximately 8GB on disk.

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <repo>
cd agentic-browser
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Pull Ollama models
```bash
ollama pull llava:13b
ollama pull llama3.1:8b-instruct-q4_0
```
Confirm both are available:
```bash
ollama list
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

---

## Running

### Start Ollama (if not already running)
```bash
ollama serve
```

### Start the API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the dashboard (separate terminal)
```bash
streamlit run dashboard/app.py
```

Dashboard opens at: http://localhost:8501
API docs at: http://localhost:8000/docs

---

## API Usage

### Run a task (streaming)
```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Research Anthropic funding rounds", "max_steps": 25}' \
  --no-buffer
```

### Check task status
```bash
curl http://localhost:8000/tasks/{task_id}
```

### Get final report
```bash
curl http://localhost:8000/tasks/{task_id}/report
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Project Structure

```
agentic-browser/
├── agent/
│   ├── loop.py       # LangGraph graph + run_task entry point
│   ├── nodes.py      # OBSERVE, THINK, ACT, VERIFY nodes
│   ├── actions.py    # browser action dispatcher
│   ├── vision.py     # llava:13b wrapper
│   ├── planner.py    # llama3.1 wrapper
│   └── models.py     # Pydantic models
├── browser/
│   ├── controller.py # Playwright browser controller
│   └── screenshot.py # screenshot capture + encoding
├── api/
│   └── main.py       # FastAPI endpoints
├── dashboard/
│   └── app.py        # Streamlit live dashboard
├── db/
│   └── storage.py    # SQLite task/report storage
├── tests/
│   └── test_vision.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Known Constraints

- llava:13b has ~60-70% UI element identification reliability.
  The agent retries with cropped regions and falls back to click_text on failures.
- Max 25 steps per task — graceful partial report on limit reached.
- Models load sequentially on RTX 2060 — expect 10-20s per step on first run.
- Headless mode is default. Set headless=False in the API request to watch the browser.
