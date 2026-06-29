"""Tests for the StepTracer — JSONL log file creation and record format."""

import json
import pytest
from agent.tracer import StepTracer


class TestStepTracer:

    def test_creates_log_file(self, tmp_path):
        tracer = StepTracer("task-trace-001", log_dir=str(tmp_path))
        tracer.close()
        log_path = tmp_path / "task-trace-001" / "trace.jsonl"
        assert log_path.exists()

    def test_log_step_writes_valid_jsonl(self, tmp_path):
        tracer = StepTracer("task-trace-002", log_dir=str(tmp_path))
        tracer.log_step(
            1, "act",
            {"action": "navigate", "target": "https://example.com", "result": "success"},
        )
        tracer.log_step(
            2, "act",
            {"action": "extract", "target": "page", "result": "some text"},
        )
        tracer.close()

        log_path = tmp_path / "task-trace-002" / "trace.jsonl"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["step"] == 1
        assert record["node"] == "act"
        assert record["action"] == "navigate"
        assert "timestamp" in record
        assert record["task_id"] == "task-trace-002"

    def test_log_step_includes_all_required_keys(self, tmp_path):
        tracer = StepTracer("task-trace-003", log_dir=str(tmp_path))
        tracer.log_step(
            5, "observe",
            {"action": "screenshot", "target": "", "result": "ok"},
        )
        tracer.close()

        log_path = tmp_path / "task-trace-003" / "trace.jsonl"
        record = json.loads(log_path.read_text().strip())
        for key in ["task_id", "step", "node", "timestamp"]:
            assert key in record

    def test_close_is_safe_to_call_twice(self, tmp_path):
        tracer = StepTracer("task-trace-004", log_dir=str(tmp_path))
        tracer.close()
        tracer.close()  # should not raise

    def test_non_serializable_data_does_not_crash(self, tmp_path):
        tracer = StepTracer("task-trace-005", log_dir=str(tmp_path))
        tracer.log_step(
            1, "act",
            {"action": object(), "target": None, "result": b"bytes"},
        )
        tracer.close()
        log_path = tmp_path / "task-trace-005" / "trace.jsonl"
        assert log_path.exists()
