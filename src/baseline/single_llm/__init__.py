"""Single-prompt baseline: the full dataset text in one SURF call."""

from src.baseline.single_llm.runner import (
    CONTEXT_WINDOWS,
    DEFAULT_MODELS,
    run_all_models,
)
from src.baseline.single_llm.schema import PANGAEA_FULL_STANDARD_NAME

__all__ = [
    "CONTEXT_WINDOWS",
    "DEFAULT_MODELS",
    "PANGAEA_FULL_STANDARD_NAME",
    "run_all_models",
]
