# 🧠 LangGraph RAG Chatbot — with UI, Evals & Tracing

A production-grade **Corrective RAG (CRAG)** chatbot with a Streamlit dashboard, RAGAS evaluation, and LangSmith tracing. Built with LangGraph, Groq (Llama-3.3-70b), and DataStax AstraDB.

---

## Architecture

```
                        ┌─────────────┐
                        │    START    │
                        └──────┬──────┘
                               │ route_question
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          [retrieve]     [wiki_search]   [arxiv_search]
               └───────────────┼───────────────┘
                               ▼
                      [grade_documents]  ◄──────────────┐
                               │                        │
              ┌────────────────┴─────────────────┐      │
              │ docs found                        │ none │
              ▼                                  ▼      │
          [generate]                   [transform_query]─┘
              │                          (max 2 retries)
              ▼
    [check_hallucination]
              │
     ┌────────┴────────┐
     │ grounded        │ hallucinated
     ▼                 ▼
[append_history]   [generate]  (max 2 retries)
     │
     ▼
   [END]
```

---

## Project Structure

```
langgraph-astradb-rag/
├── rag_pipeline.py      ← Core graph — imported by app.py and eval.py
├── app.py               ← Streamlit dashboard UI
├── eval.py              ← RAGAS + LangSmith evaluation script
├── requirements.txt     ← All dependencies
├── .env.example         ← Template — copy to .env
├── .env                 ← Your secrets (git-ignored)
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/langgraph-astradb-rag.git
cd langgraph-astradb-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env — fill in ASTRA_DB_TOKEN, ASTRA_DB_ID, GROQ_API_KEY
# Add LANGCHAIN_API_KEY for LangSmith tracing
# Add OPENAI_API_KEY if you want RAGAS scoring
```

### 3. Ingest documents (first time only)

```python
from rag_pipeline import ingest_documents
ingest_documents()
```

Or run from the terminal:

```bash
python -c "from rag_pipeline import ingest_documents; ingest_documents()"
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 5. Run evaluations

```bash
python eval.py
```

Outputs a `eval_report_<timestamp>.json` (custom metrics) and optionally a `ragas_scores_<timestamp>.json`.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `ASTRA_DB_TOKEN` | ✅ | AstraDB authentication token (`AstraCS:...`) |
| `ASTRA_DB_ID` | ✅ | AstraDB database UUID |
| `GROQ_API_KEY` | ✅ | Groq LLM API key |
| `LANGCHAIN_API_KEY` | Recommended | LangSmith tracing |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name (default: `langgraph-astradb-rag`) |
| `OPENAI_API_KEY` | For RAGAS only | RAGAS uses GPT for faithfulness/relevancy scoring |

---

## Features

### Dashboard (`app.py`)
- **Chat interface** — multi-turn conversation with full history
- **Live graph trace** — sidebar shows every node executed for the last query
- **Source citations** — retrieved document snippets shown in sidebar
- **Hallucination status** — grounded/hallucinated indicator per answer
- **Session metrics** — query count and grounded-answer percentage

### Evaluation (`eval.py`)
- **Custom metrics** — hallucination rate, avg answer length, docs kept ratio, route distribution
- **RAGAS metrics** — faithfulness, answer relevancy, context precision, context recall
- **LangSmith** — all runs are automatically traced when `LANGCHAIN_TRACING_V2=true`
- Reports saved as timestamped JSON files

### Pipeline (`rag_pipeline.py`)
- Multi-source routing (vectorstore / Wikipedia / arXiv)
- Corrective RAG — per-document relevance grading
- Query rewriting with retry loop (max 2 retries)
- Hallucination detection with retry cap
- Multi-turn chat memory via `chat_history`
- `trace` field in state for UI visibility

---

## Tech Stack

| Component | Library / Service |
|---|---|
| Graph orchestration | LangGraph |
| LLM | Groq — Llama-3.3-70b-versatile |
| Vector store | DataStax AstraDB via cassio |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| UI | Streamlit |
| Evaluation | RAGAS + LangSmith |
| Wikipedia | langchain_community WikipediaQueryRun |
| arXiv | langchain_community ArxivQueryRun |

---

## LangSmith Tracing

Every run is automatically sent to LangSmith when `LANGCHAIN_TRACING_V2=true`. You can:
- See the full node-by-node execution trace
- Inspect inputs/outputs at each node
- Build evaluation datasets from production queries
- Compare runs across model versions

Dashboard: **https://smith.langchain.com**

---

## RAGAS Metrics Explained

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer supported by the retrieved context? |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the retrieved docs actually relevant? |
| `context_recall` | Did retrieval capture all necessary information? |

> Note: RAGAS requires `OPENAI_API_KEY` as it uses GPT internally for scoring.

---

## License

MIT
