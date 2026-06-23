"""Trace tool calls made during plan execution."""

from copy import deepcopy
from typing import Any, Dict, List, Optional


class ToolTrace:
    """
    Stores tool call records for a single plan execution.

    Each record captures which tool ran, which agent called it, the tool input,
    and the output or error returned by the tool.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record_tool_call(
        self,
        tool: Optional[str],
        agent: str,
        tool_input: Dict[str, Any],
        output: Any,
    ) -> None:
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
        return [record["tool"] for record in self._records if record.get("tool")]

    def find_tool_calls(self, tool: str) -> List[Dict[str, Any]]:
        """Return agent, input, and output records for calls to the given tool."""
        return [
            {
                "agent": record["agent"],
                "input": deepcopy(record["input"]),
                "output": deepcopy(record["output"]),
            }
            for record in self._records
            if record.get("tool") == tool
        ]
