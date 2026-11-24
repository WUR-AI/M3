"""Utilities for building Pydantic models and prompt templates from field specs."""

from typing import Any, Dict, List, Optional, get_args, get_origin

from pydantic import BaseModel, Field, create_model


def to_model_name(standard_name: str) -> str:
    return "".join(part.capitalize() for part in standard_name.split("_")) + "Metadata"


def _model_ref_name(type_ref: Any) -> Optional[str]:
    if isinstance(type_ref, str):
        return type_ref
    forward_arg = getattr(type_ref, "__forward_arg__", None)
    if forward_arg is not None:
        return forward_arg
    return None


def resolve_type(
    type_hint: Any,
    model_registry: Optional[Dict[str, type[BaseModel]]] = None,
) -> Any:
    """Resolve forward references like List['FileSet'] using a model registry."""
    model_registry = model_registry or {}

    ref_name = _model_ref_name(type_hint)
    if ref_name is not None:
        if ref_name not in model_registry:
            raise KeyError(f"Unknown nested model reference: {ref_name}")
        return model_registry[ref_name]

    origin = get_origin(type_hint)
    if origin is list:
        inner_args = get_args(type_hint)
        if not inner_args:
            return type_hint
        return List[resolve_type(inner_args[0], model_registry)]

    if origin is not None:
        inner_args = tuple(
            resolve_type(arg, model_registry) for arg in get_args(type_hint)
        )
        return origin[inner_args]

    return type_hint


def build_schema_for_standard(
    standard_name: str,
    field_spec: Dict[str, Dict[str, Any]],
    *,
    model_name: Optional[str] = None,
    model_registry: Optional[Dict[str, type[BaseModel]]] = None,
) -> type[BaseModel]:
    model_fields: Dict[str, Any] = {}
    for field_name, spec in field_spec.items():
        field_type = resolve_type(spec["type"], model_registry)
        model_fields[field_name] = (
            field_type,
            Field(default=spec["default"], description=spec["description"]),
        )
    return create_model(model_name or to_model_name(standard_name), **model_fields)


def build_prompt_template(field_spec: Dict[str, Dict[str, Any]]) -> str:
    lines = ["{"]
    entries = list(field_spec.items())
    for index, (field_name, spec) in enumerate(entries):
        prompt_hint = spec.get("prompt_hint", "...")
        comma = "," if index < len(entries) - 1 else ""
        lines.append(f'    "{field_name}": "{prompt_hint}"{comma}')
    lines.append("}")
    return "\n".join(lines)
