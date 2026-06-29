"""SQLite persistence layer — schema init and CRUD helpers for tasks, steps, and reports."""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "db/agentic_browser.db"

CREATE_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    max_steps INTEGER DEFAULT 25,
    steps_taken INTEGER DEFAULT 0
);
"""

CREATE_STEPS_TABLE = """
CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id),
    step_number INTEGER,
    screenshot_path TEXT,
    llava_output TEXT,
    llama_reasoning TEXT,
    action_taken TEXT,
    action_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id),
    goal TEXT,
    status TEXT,
    findings_json TEXT,
    agent_trace_json TEXT,
    screenshot_paths_json TEXT,
    discovered_facts_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


async def init_db() -> None:
    """Create all tables if they do not already exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_TASKS_TABLE)
        await db.execute(CREATE_STEPS_TABLE)
        await db.execute(CREATE_REPORTS_TABLE)
        try:
            # migration for databases created before discovered_facts_json existed
            await db.execute("ALTER TABLE reports ADD COLUMN discovered_facts_json TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


async def save_task(task_id: str, goal: str, max_steps: int) -> None:
    """Insert a new task row with status 'running'."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tasks (id, goal, max_steps, status) VALUES (?, ?, ?, 'running')",
            (task_id, goal, max_steps),
        )
        await db.commit()
    logger.info("Saved task %s", task_id)


async def update_task_status(task_id: str, status: str, steps_taken: int) -> None:
    """Update a task's status, step count, and updated_at timestamp."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET status=?, steps_taken=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=?",
            (status, steps_taken, task_id),
        )
        await db.commit()
    logger.info("Updated task %s → status=%s steps_taken=%d", task_id, status, steps_taken)


async def save_step(task_id: str, step_record: dict) -> None:
    """Insert a step row. The target is folded into action_taken as 'action:target'."""
    action_taken = step_record.get("action_taken", "")
    target = step_record.get("target", "")
    combined_action = f"{action_taken}:{target}"
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO steps (id, task_id, step_number, action_taken, action_result) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                task_id,
                step_record.get("step_number"),
                combined_action,
                step_record.get("action_result"),
            ),
        )
        await db.commit()


async def save_report(task_id: str, report: dict) -> None:
    """Insert a report row, JSON-serializing the nested findings/trace/paths."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO reports "
            "(id, task_id, goal, status, findings_json, agent_trace_json, "
            "screenshot_paths_json, discovered_facts_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                task_id,
                report.get("goal", ""),
                report.get("status", ""),
                json.dumps(report.get("findings", {})),
                json.dumps(report.get("agent_reasoning_trace", [])),
                json.dumps(report.get("screenshot_paths", [])),
                json.dumps(report.get("discovered_facts", [])),
            ),
        )
        await db.commit()
    logger.info("Saved report for task %s", task_id)
