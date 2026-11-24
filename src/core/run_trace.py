"""Durable, run-scoped record of model and tool inputs and outputs."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def message_record(message: Any) -> dict[str, Any]:
    """Keep the complete message sent to or returned by a model."""
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True)
    return {"type": type(message).__name__, "content": str(message)}


class RunTraceRecorder:
    """Append each event immediately so a failed run still has a useful trace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def create(cls, context_name: str) -> "RunTraceRecorder":
        root = Path(__file__).resolve().parents[2]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", context_name).strip("._") or "context"
        filename = f"{safe_name[:80]}_{uuid.uuid4().hex}.jsonl"
        return cls(root / "outputs" / "traces" / filename)

    def record(self, event: str, **data: Any) -> None:
        entry = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
