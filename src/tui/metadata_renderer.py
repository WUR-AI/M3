"""
Metadata renderer for the TUI.

Builds a rounded-corner tree view of final metadata as Rich markup strings
(colors defined in palette.py) so Textual can render them directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tui.palette import (
    PLAN_MUTED_COLOR,
    PLAN_STEP_COLOR,
    PLAN_TITLE_COLOR,
    PLAN_TREE_GUIDE_COLOR,
    PLAN_VALUE_COLOR,
)

_METADATA_KEY_COLOR = PLAN_STEP_COLOR


def format_metadata(
    metadata: Dict[str, Any],
    *,
    title: str = "Final Metadata",
    width: Optional[int] = None,
) -> str:
    """
    Render metadata as a colored, rounded-corner tree (Rich markup) for the TUI.

    Args:
        metadata: Final metadata dictionary from the pipeline.
        title: Heading shown at the tree root.
        width: Reserved for layout; values are never truncated.
    """
    _ = width  # kept for API parity with format_plan
    if not metadata:
        return (
            f"{_guide('╭─ ')}{_style(PLAN_TITLE_COLOR, title)}\n"
            f"{_guide('╰─ ')}{_style(PLAN_MUTED_COLOR, 'No metadata fields.')}"
        )

    lines = [f"{_guide('╭─ ')}{_style(PLAN_TITLE_COLOR, title)}"]
    items = list(metadata.items())
    for index, (key, value) in enumerate(items):
        is_last = index == len(items) - 1
        lines.extend(
            _format_entry_lines(
                str(key),
                value,
                prefix="",
                is_last=is_last,
            )
        )
    return "\n".join(lines)


def _format_entry_lines(
    key: str,
    value: Any,
    *,
    prefix: str,
    is_last: bool,
) -> List[str]:
    branch = "╰─ " if is_last else "├─ "
    cont = "    " if is_last else "│   "

    if _is_scalar(value):
        return _format_scalar_lines(key, value, prefix=prefix, branch=branch, cont=cont)

    if isinstance(value, dict):
        lines = [
            _guide(prefix + branch) + _style(_METADATA_KEY_COLOR, f"{key}:")
        ]
        nested = list(value.items())
        for index, (nested_key, nested_value) in enumerate(nested):
            lines.extend(
                _format_entry_lines(
                    str(nested_key),
                    nested_value,
                    prefix=prefix + cont,
                    is_last=index == len(nested) - 1,
                )
            )
        return lines

    if isinstance(value, list):
        if not value:
            return [
                _guide(prefix + branch)
                + _style(_METADATA_KEY_COLOR, f"{key}:")
                + " "
                + _style(PLAN_MUTED_COLOR, "(empty)")
            ]
        if all(_is_scalar(item) for item in value):
            return [
                _guide(prefix + branch)
                + _labeled(key, _join_scalars(value))
            ]
        lines = [
            _guide(prefix + branch) + _style(_METADATA_KEY_COLOR, f"{key}:")
        ]
        for index, item in enumerate(value):
            item_key = f"[{index}]"
            lines.extend(
                _format_entry_lines(
                    item_key,
                    item,
                    prefix=prefix + cont,
                    is_last=index == len(value) - 1,
                )
            )
        return lines

    return _format_scalar_lines(
        key, value, prefix=prefix, branch=branch, cont=cont
    )


def _format_scalar_lines(
    key: str,
    value: Any,
    *,
    prefix: str,
    branch: str,
    cont: str,
) -> List[str]:
    text = _scalar_repr(value)
    parts = text.split("\n")
    lines = [_guide(prefix + branch) + _labeled(key, parts[0])]
    for part in parts[1:]:
        lines.append(
            _guide(prefix + cont + "   ") + _style(PLAN_VALUE_COLOR, part)
        )
    return lines


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _scalar_repr(value: Any) -> str:
    if value is None:
        return "(null)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)


def _join_scalars(items: List[Any]) -> str:
    return ", ".join(_scalar_repr(item) for item in items)


def _guide(text: str) -> str:
    return _style(PLAN_TREE_GUIDE_COLOR, text)


def _style(color: str, text: str) -> str:
    return f"[{color}]{text}[/]"


def _labeled(label: str, value: str) -> str:
    label_markup = _style(_METADATA_KEY_COLOR, f"{label}:")
    if not value:
        return label_markup
    return f"{label_markup} {_style(PLAN_VALUE_COLOR, value)}"
