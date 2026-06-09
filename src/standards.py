"""
This file contains predefined, named metadata standards that can be used by the orchestrator.

Each standard has:
1. A string template (METADATA_STANDARDS) - used for prompting the LLM
2. A Pydantic model (METADATA_SCHEMAS) - used for structured output validation
"""

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, create_model


# =============================================================================
# SINGLE SOURCE OF TRUTH
# Define each standard once, then derive both prompt template and Pydantic schema.
# =============================================================================

STANDARD_DEFINITIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "spatial_ecological": {
        "title": {
            "type": str,
            "default": ...,
            "description": "Title of the dataset",
            "prompt_hint": "...",
        },
        "description": {
            "type": str,
            "default": ...,
            "description": "Description of the dataset",
            "prompt_hint": "...",
        },
        "subject": {
            "type": Optional[str],
            "default": None,
            "description": "Subject/topic",
            "prompt_hint": "...",
        },
        "spatial_coverage": {
            "type": Optional[Dict[str, float]],
            "default": None,
            "description": (
                "Geographic bounding box with keys: "
                "min_lat, min_lon, max_lat, max_lon"
            ),
            # Keep the exact prompt wording requested.
            "prompt_hint": (
                "Geographic bounding box in WGS84 with numeric fields: "
                "min_lat, min_lon, max_lat, max_lon"
            ),
        },
        "spatial_resolution": {
            "type": Optional[str],
            "default": None,
            "description": "Spatial resolution of the data",
            "prompt_hint": "...",
        },
        "temporal_coverage": {
            "type": Optional[str],
            "default": None,
            "description": "Time period covered, from and to date",
            "prompt_hint": "Time period covered, from and to date",
        },
        "temporal_resolution": {
            "type": Optional[str],
            "default": None,
            "description": "Temporal resolution of the data",
            "prompt_hint": "Temporal resolution of the data, e.g. daily, monthly, yearly",
        },
        "methods": {
            "type": Optional[str],
            "default": None,
            "description": "Methods used for data collection",
            "prompt_hint": "Methods used for data collection",
        },
        "format": {
            "type": Optional[str],
            "default": None,
            "description": "Data format",
            "prompt_hint": "...",
        },
    },
    # Dummy standard for testing @standard multi-selection behavior in TUI.
    "dummy_standard": {
        "title": {
            "type": str,
            "default": ...,
            "description": "Dummy title field",
            "prompt_hint": "...",
        },
        "summary": {
            "type": Optional[str],
            "default": None,
            "description": "Dummy summary field",
            "prompt_hint": "...",
        },
        "owner": {
            "type": Optional[str],
            "default": None,
            "description": "Dummy owner field",
            "prompt_hint": "...",
        },
        "version": {
            "type": Optional[str],
            "default": None,
            "description": "Dummy version field",
            "prompt_hint": "...",
        },
    },
}


def _to_model_name(standard_name: str) -> str:
    return "".join(part.capitalize() for part in standard_name.split("_")) + "Metadata"


def _build_schema_for_standard(
    standard_name: str,
    field_spec: Dict[str, Dict[str, Any]],
) -> type[BaseModel]:
    model_fields: Dict[str, Any] = {}
    for field_name, spec in field_spec.items():
        model_fields[field_name] = (
            spec["type"],
            Field(default=spec["default"], description=spec["description"]),
        )
    return create_model(_to_model_name(standard_name), **model_fields)


def _build_prompt_template(field_spec: Dict[str, Dict[str, Any]]) -> str:
    lines = ["{"]
    entries = list(field_spec.items())
    for index, (field_name, spec) in enumerate(entries):
        prompt_hint = spec.get("prompt_hint", "...")
        comma = "," if index < len(entries) - 1 else ""
        lines.append(f'    "{field_name}": "{prompt_hint}"{comma}')
    lines.append("}")
    return "\n".join(lines)


# =============================================================================
# SCHEMA REGISTRY - Maps standard names to Pydantic models
# =============================================================================

METADATA_SCHEMAS: Dict[str, type[BaseModel]] = {
    standard_name: _build_schema_for_standard(standard_name, field_spec)
    for standard_name, field_spec in STANDARD_DEFINITIONS.items()
}


def get_schema_for_standard(standard_name: str) -> Optional[type[BaseModel]]:
    """
    Get the Pydantic schema class for a given standard name.
    
    Args:
        standard_name: Name of the metadata standard
        
    Returns:
        Pydantic model class, or None if not found
    """
    return METADATA_SCHEMAS.get(standard_name)

METADATA_STANDARDS = {
    standard_name: _build_prompt_template(field_spec)
    for standard_name, field_spec in STANDARD_DEFINITIONS.items()
}


def load_metadata_standard(standard_arg: str) -> str:
    """
    Load metadata standard content from the registry or a file path.
    """
    if standard_arg in METADATA_STANDARDS:
        return METADATA_STANDARDS[standard_arg]

    standard_path = Path(standard_arg)
    if standard_path.exists():
        return standard_path.read_text()

    raise ValueError(
        f"Metadata standard '{standard_arg}' not found as a predefined standard or as a valid file path."
    )
