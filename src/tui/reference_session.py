"""
Session state for resolved @path references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .references import ResolvedReference, collect_verified_references


@dataclass
class ReferenceSessionState:
    current_input_refs: list[ResolvedReference] = field(default_factory=list)
    last_submitted_refs: list[ResolvedReference] = field(default_factory=list)
    history_refs: list[list[ResolvedReference]] = field(default_factory=list)

    def update_from_input(self, text: str, root: Path) -> None:
        self.current_input_refs = collect_verified_references(text, root)

    def commit_current_input(self, text: str, root: Path) -> list[ResolvedReference]:
        self.last_submitted_refs = collect_verified_references(text, root)
        self.history_refs.append(self.last_submitted_refs.copy())
        return self.last_submitted_refs

    def clear_current(self) -> None:
        self.current_input_refs = []

