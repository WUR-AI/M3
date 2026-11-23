"""
Plan renderer for the TUI.

Builds a rounded-corner tree view of execution plans as Rich markup strings
(colors defined in palette.py) so Textual can render them directly.
"""

from __future__ import annotations

import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.schemas import Plan
from src.tui.palette import (
    PLAN_LABEL_COLOR,
    PLAN_MUTED_COLOR,
    PLAN_STEP_COLOR,
    PLAN_TITLE_COLOR,
    PLAN_TREE_GUIDE_COLOR,
    PLAN_VALUE_COLOR,
)

_MARKUP_PATTERN = re.compile(r"\[[^\]]*\]")


def format_plan(
    plan: Plan,
    *,
    title: str = "Execution Plan",
    width: Optional[int] = None,
) -> str:
    """
    Render a Plan as a colored, rounded-corner tree (Rich markup) for the TUI.

    Args:
        plan: The generated execution plan.
        title: Heading shown at the tree root.
        width: Max line width (defaults to terminal width minus padding).
    """
    render_width = _resolve_width(width)
    steps = plan.to_dict_list()
    if not steps:
        return (
            f"{_guide('╭─ ')}{_style(PLAN_TITLE_COLOR, title)}\n"
            f"{_guide('╰─ ')}{_style(PLAN_MUTED_COLOR, 'No steps in plan.')}"
        )

    lines = [f"{_guide('╭─ ')}{_style(PLAN_TITLE_COLOR, title)}"]
    for index, step in enumerate(steps):
        is_last_step = index == len(steps) - 1
        lines.extend(_format_step_lines(index + 1, step, is_last_step, render_width))
    return "\n".join(lines)


def _resolve_width(width: Optional[int]) -> int:
    if width is not None and width > 0:
        return max(40, width)
    try:
        columns = shutil.get_terminal_size().columns
    except OSError:
        columns = 80
    return max(40, columns - 6)


def _format_step_lines(
    index: int,
    step: Dict[str, Any],
    is_last_step: bool,
    width: int,
) -> List[str]:
    task = step.get("task", "<unknown>")
    player = step.get("player", "<unknown>")
    target_resources = step.get("target_resources", []) or []
    inputs: Dict[str, str] = step.get("inputs", {}) or {}
    outputs: List[str] = step.get("outputs", []) or []
    rationale = (step.get("rationale") or "").strip()

    text_budget = max(24, width - 14)
    step_branch = "╰─ " if is_last_step else "├─ "
    step_cont = "    " if is_last_step else "│   "

    lines = [
        _guide(step_branch)
        + _style(PLAN_STEP_COLOR, f"Step {index}: {_fit(task, text_budget)}")
    ]

    # Each entry: (rendered content, optional child lines)
    details: List[Tuple[str, List[str]]] = [
        (_labeled("Player", player, text_budget - 8), []),
    ]
    if target_resources:
        details.append(
            (
                _labeled("Targets", _join(target_resources), text_budget - 9),
                [],
            )
        )
    if inputs:
        input_children = [
            _style(PLAN_VALUE_COLOR, _fit(f"{param} <- {artifact}", text_budget - 4))
            for param, artifact in inputs.items()
        ]
        details.append((_style(PLAN_LABEL_COLOR, "Inputs:"), input_children))
    if outputs:
        details.append(
            (
                _labeled("Outputs", _join(outputs), text_budget - 9),
                [],
            )
        )
    if rationale:
        details.append(
            (
                f"{_style(PLAN_MUTED_COLOR, 'Rationale:')} "
                f"{_style(PLAN_VALUE_COLOR, _fit(rationale, text_budget - 11))}",
                [],
            )
        )

    for detail_index, (content, children) in enumerate(details):
        is_last_detail = detail_index == len(details) - 1
        detail_branch = "╰─ " if is_last_detail else "├─ "
        lines.append(_guide(step_cont + detail_branch) + content)

        if children:
            child_cont = step_cont + ("    " if is_last_detail else "│   ")
            for child_index, child_content in enumerate(children):
                is_last_child = child_index == len(children) - 1
                child_branch = "╰─ " if is_last_child else "├─ "
                lines.append(_guide(child_cont + child_branch) + child_content)

    return lines


def _guide(text: str) -> str:
    return _style(PLAN_TREE_GUIDE_COLOR, text)


def _style(color: str, text: str) -> str:
    return f"[{color}]{text}[/]"


def _labeled(label: str, value: str, budget: int) -> str:
    label_markup = _style(PLAN_LABEL_COLOR, f"{label}:")
    if not value:
        return label_markup
    return f"{label_markup} {_style(PLAN_VALUE_COLOR, _fit(value, budget))}"


def _fit(text: str, max_width: int) -> str:
    plain = _strip_markup(text)
    if max_width < 4 or len(plain) <= max_width:
        return text
    return plain[: max_width - 3].rstrip() + "..."


def _strip_markup(text: str) -> str:
    return _MARKUP_PATTERN.sub("", text)


def _join(items: Iterable[str]) -> str:
    return ", ".join(str(item) for item in items)
