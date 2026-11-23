"""
Reference parsing and resolution utilities for @path tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from rich.text import Text

REFERENCE_PATTERN = re.compile(r"@([^\s@]+)")


@dataclass(frozen=True)
class ResolvedReference:
    raw_token: str
    path_name: str
    absolute_path: Path
    exists: bool
    is_file: bool
    is_dir: bool


@dataclass(frozen=True)
class AtSuggestion:
    token: str
    kind: str  # "file" | "standard"


def list_visible_paths(root: Path) -> list[str]:
    visible_paths: list[str] = []

    def walk(directory: Path, prefix: str = "") -> None:
        for entry in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            if entry.name.startswith("."):
                continue

            relative_name = f"{prefix}{entry.name}"
            visible_paths.append(relative_name)

            if entry.is_dir():
                walk(entry, f"{relative_name}/")

    walk(root)
    return visible_paths


def get_at_matches(active_token: str, root: Path) -> list[str]:
    if not active_token.startswith("@"):
        return []

    file_prefix = active_token[1:]
    return [
        path_name
        for path_name in list_visible_paths(root)
        if not file_prefix or path_name.lower().startswith(file_prefix.lower())
    ]


def get_standard_matches(active_token: str, standard_names: list[str]) -> list[str]:
    if not active_token.startswith("@"):
        return []

    standard_prefix = active_token[1:].lower()
    return [
        standard_name
        for standard_name in sorted(standard_names)
        if not standard_prefix or standard_name.lower().startswith(standard_prefix)
    ]


def extract_standard_mentions(text: str, standard_names: list[str]) -> list[str]:
    found: list[str] = []
    standard_set = set(standard_names)
    for match in REFERENCE_PATTERN.finditer(text):
        token = match.group(1)
        if token in standard_set and token not in found:
            found.append(token)
    return found


def extract_invalid_standard_mentions(
    text: str,
    *,
    standard_names: list[str],
    root: Path,
) -> list[str]:
    invalid: list[str] = []
    standard_set = set(standard_names)
    for match in REFERENCE_PATTERN.finditer(text):
        token = match.group(1)
        if token in standard_set:
            continue
        if resolve_reference(token, root).exists:
            continue
        if token not in invalid:
            invalid.append(token)
    return invalid


def get_at_suggestions(
    active_token: str,
    *,
    root: Path,
    standard_names: list[str],
) -> list[AtSuggestion]:
    file_matches = [AtSuggestion(token=match, kind="file") for match in get_at_matches(active_token, root)]
    standard_matches = [
        AtSuggestion(token=match, kind="standard")
        for match in get_standard_matches(active_token, standard_names)
    ]
    return standard_matches + file_matches


def resolve_reference(path_name: str, root: Path) -> ResolvedReference:
    absolute_path = (root / path_name).resolve()
    exists = absolute_path.exists()
    return ResolvedReference(
        raw_token=f"@{path_name}",
        path_name=path_name,
        absolute_path=absolute_path,
        exists=exists,
        is_file=absolute_path.is_file() if exists else False,
        is_dir=absolute_path.is_dir() if exists else False,
    )


def collect_verified_references(text: str, root: Path) -> list[ResolvedReference]:
    references: list[ResolvedReference] = []
    for match in REFERENCE_PATTERN.finditer(text):
        reference = resolve_reference(match.group(1), root)
        if reference.exists:
            references.append(reference)
    return references


def stylize_verified_references(
    rendered: Text,
    *,
    root: Path,
    standard_names: list[str],
    text_color: str,
    background_color: str,
    standard_text_color: str,
    standard_background_color: str,
) -> Text:
    for match in REFERENCE_PATTERN.finditer(rendered.plain):
        token = match.group(1)
        if token in standard_names:
            start, end = match.span()
            rendered.stylize(
                f"{standard_text_color} on {standard_background_color}",
                start,
                end,
            )
            continue

        reference = resolve_reference(token, root)
        if reference.exists:
            start, end = match.span()
            rendered.stylize(
                f"{text_color} on {background_color}",
                start,
                end,
            )
    return rendered

