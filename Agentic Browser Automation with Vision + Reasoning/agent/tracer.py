"""Step tracer — per-step JSONL trace files consumed by the API and dashboard."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("agent.tracer")


class StepTracer:
    """Appends one JSON record per line to logs/{task_id}/trace.jsonl."""

    def __init__(self, task_id: str, log_dir: str = "logs"):
        self.task_id = task_id
        trace_dir = Path(log_dir) / task_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(trace_dir / "trace.jsonl", "a", encoding="utf-8")

    def log_step(self, step_number: int, node: str, data: dict) -> None:
        """Write one trace record as a single JSON line and flush immediately."""
        try:
            record = {
                "task_id": self.task_id,
                "step": step_number,
                "node": node,
                "timestamp": datetime.utcnow().isoformat(),
                **data,
            }
            self.log_file.write(json.dumps(record, default=str) + "\n")
            self.log_file.flush()
        except Exception as exc:  # noqa: BLE001 — tracing must never break the agent loop
            logger.error("log_step failed for task %s: %s", self.task_id, exc)

    def close(self) -> None:
        try:
            self.log_file.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("close failed for task %s: %s", self.task_id, exc)
