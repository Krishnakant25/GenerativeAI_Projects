"""FastAPI REST API — trigger agent tasks, stream progress, and fetch reports."""

import asyncio
import logging
import os
import uuid
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiosqlite

from agent.loop import run_task
from db.storage import init_db, DB_PATH

logger = logging.getLogger("api.main")

app = FastAPI(title="Agentic Browser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialised")


class TaskRequest(BaseModel):
    goal: str
    max_steps: int = 25
    headless: bool = True


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


@app.post("/tasks")
async def create_task(request: TaskRequest):
    """Start a new agent task and stream progress as newline-delimited JSON."""
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=400, detail="goal cannot be empty")

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            async for event in run_task(
                request.goal, request.max_steps, request.headless
            ):
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception as exc:  # noqa: BLE001 — surface as a stream error event
            logger.error("POST /tasks stream failed: %s", exc)
            yield (json.dumps({"event": "error", "error": str(exc)}) + "\n").encode(
                "utf-8"
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Return the current status of a task from the database."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            )
            row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="task not found")

        return JSONResponse(
            {
                "task_id": row["id"],
                "goal": row["goal"],
                "status": row["status"],
                "steps_taken": row["steps_taken"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /tasks/%s failed: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks/{task_id}/report")
async def get_report(task_id: str):
    """Return the final report for a completed task."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM reports WHERE task_id = ?", (task_id,)
            )
            row = await cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404, detail="report not found for this task"
            )

        return JSONResponse(
            {
                "task_id": task_id,
                "goal": row["goal"],
                "status": row["status"],
                "findings": _safe_json_load(row["findings_json"]),
                "agent_trace": _safe_json_load(row["agent_trace_json"]),
                "screenshot_paths": _safe_json_load(row["screenshot_paths_json"]),
                "discovered_facts": _safe_json_load(row["discovered_facts_json"]),
                "created_at": row["created_at"],
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /tasks/%s/report failed: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks/{task_id}/trace")
async def get_trace(task_id: str):
    """Return the per-step JSONL trace written by StepTracer as a JSON list."""
    trace_path = Path("logs") / task_id / "trace.jsonl"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="trace not found for this task")

    try:
        steps = []
        with open(trace_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
        return JSONResponse({"task_id": task_id, "steps": steps})
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /tasks/%s/trace failed: %s", task_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Simple health check echoing the configured Ollama models."""
    return JSONResponse(
        {
            "status": "ok",
            "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "vision_model": os.getenv("VISION_MODEL", "llava:13b"),
            "reasoning_model": os.getenv(
                "REASONING_MODEL", "llama3.1:8b-instruct-q4_0"
            ),
        }
    )


def _safe_json_load(raw):
    """Parse a JSON string, returning the raw value unchanged on parse failure."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
