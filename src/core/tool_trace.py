"""Trace tool calls made during plan execution."""

import json
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, List, Optional


def tool_result_is_error(result: Any) -> bool:
    """Recognize tool errors returned as dictionaries or serialized JSON."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return result.strip().lower().startswith("error:")
    return isinstance(result, dict) and "error" in result


class ToolTrace:
    """
    Stores tool call records for a single plan execution.

    Each record captures which tool ran, which agent called it, the tool input,
    and the output or error returned by the tool.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._lock = Lock()

    def record_tool_call(
        self,
        tool: Optional[str],
        agent: str,
        tool_input: Dict[str, Any],
        output: Any,
    ) -> None:
        with self._lock:
            self._records.append(
                {
                    "tool": tool,
                    "agent": agent,
                    "input": deepcopy(tool_input),
                    "output": deepcopy(output),
                }
            )

    def list_tools_called(self) -> List[str]:
        """Return tool names in the order they were called."""
        with self._lock:
            return [record["tool"] for record in self._records if record.get("tool")]

    def find_tool_calls(self, tool: str) -> List[Dict[str, Any]]:
        """Return agent, input, and output records for calls to the given tool."""
        with self._lock:
            return [
                {
                    "agent": record["agent"],
                    "input": deepcopy(record["input"]),
                    "output": deepcopy(record["output"]),
                }
                for record in self._records
                if record.get("tool") == tool
            ]
