"""Local-only audit log for compliance. Never leaves the device."""
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("copilot.audit")

AUDIT_DIR = "screen_copilot/audit"

class AuditLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        os.makedirs(AUDIT_DIR, exist_ok=True)
        self.log_path = os.path.join(AUDIT_DIR, f"audit_{session_id}.jsonl")

    def log_event(self, event_type: str, details: dict) -> None:
        """Record an auditable event. All processing is local."""
        try:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": self.session_id,
                "event_type": event_type,  # capture | analysis | selection | blacklist_skip | suggestion
                "processed_locally": True,
                "data_sent_externally": False,
                **details
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("audit log failed: %s", e)

    def get_summary(self) -> dict:
        """Return counts of each event type for this session."""
        counts = {}
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        et = rec.get("event_type", "unknown")
                        counts[et] = counts.get(et, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return counts
