"""Metadata fields for the one-call PANGAEA baseline.

This schema is local to the baseline so the orchestrator's existing PANGAEA
standard and routing remain unchanged.
"""

from src.standards.croissant import (
    pangaea_descriptive_technical_standard,
)
from src.standards.schema_builder import build_schema_for_standard


PANGAEA_FULL_STANDARD_NAME = "single_llm_pangaea_standard"
PANGAEA_FULL_FIELDS = pangaea_descriptive_technical_standard
PangaeaSingleLLMMetadata = build_schema_for_standard(
    PANGAEA_FULL_STANDARD_NAME,
    PANGAEA_FULL_FIELDS,
    model_name="PangaeaSingleLLMMetadata",
)
