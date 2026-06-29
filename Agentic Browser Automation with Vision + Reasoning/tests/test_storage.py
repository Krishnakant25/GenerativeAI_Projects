"""Tests for the storage layer — schema init and CRUD helpers."""

import json
import os
import pytest
import aiosqlite
from db.storage import init_db, save_task, save_step, save_report, update_task_status

TEST_DB = "db/test_agentic_browser.db"


@pytest.fixture(autouse=True)
async def setup_test_db(monkeypatch):
    monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
    await init_db()
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestStorage:

    @pytest.mark.asyncio
    async def test_save_and_retrieve_task(self, monkeypatch):
        monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
        await save_task("task-001", "test goal", 25)
        async with aiosqlite.connect(TEST_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE id=?", ("task-001",)
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["goal"] == "test goal"
        assert row["status"] == "running"
        assert row["max_steps"] == 25

    @pytest.mark.asyncio
    async def test_update_task_status(self, monkeypatch):
        monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
        await save_task("task-002", "another goal", 10)
        await update_task_status("task-002", "complete", 7)
        async with aiosqlite.connect(TEST_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tasks WHERE id=?", ("task-002",)
            ) as cur:
                row = await cur.fetchone()
        assert row["status"] == "complete"
        assert row["steps_taken"] == 7

    @pytest.mark.asyncio
    async def test_save_step(self, monkeypatch):
        monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
        await save_task("task-003", "goal", 25)
        step_record = {
            "step_number": 1,
            "action_taken": "navigate",
            "target": "https://example.com",
            "action_result": "success",
        }
        await save_step("task-003", step_record)
        async with aiosqlite.connect(TEST_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM steps WHERE task_id=?", ("task-003",)
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert "navigate" in row["action_taken"]

    @pytest.mark.asyncio
    async def test_save_report(self, monkeypatch):
        monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
        await save_task("task-004", "report goal", 25)
        report = {
            "goal": "report goal",
            "status": "complete",
            "findings": {"summary": "Found some data"},
            "agent_reasoning_trace": [],
            "screenshot_paths": [],
            "discovered_facts": ["Fact 1", "Fact 2"],
        }
        await save_report("task-004", report)
        async with aiosqlite.connect(TEST_DB) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM reports WHERE task_id=?", ("task-004",)
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["status"] == "complete"
        facts = json.loads(row["discovered_facts_json"])
        assert "Fact 1" in facts

    @pytest.mark.asyncio
    async def test_init_db_is_idempotent(self, monkeypatch):
        monkeypatch.setattr("db.storage.DB_PATH", TEST_DB)
        # Running init_db twice should not raise
        await init_db()
        await init_db()
