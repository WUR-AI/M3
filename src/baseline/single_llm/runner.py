"""One SURF call per dataset: the full CSV text, or no call at all.

Context lengths are hardcoded from the model configs fetched on 2026-09-30.
SURF ``/models`` does not return ``max_model_len``, and chat probes returned
502 while the models were unloaded. The values below are the published
windows: Qwen3.6 and Qwen3.5 ``max_position_embeddings`` 262144, Devstral's
published 256k window, Mistral-Small-3.2 ``max_position_embeddings`` 131072,
and the Llama-70B window behind the ``default-text-large`` alias, 131072.
"""

from __future__ import annotations

import json
import hashlib
import math
import multiprocessing
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, get_origin

import tiktoken
from pydantic import ValidationError

from src.baseline.autoddg.llm.surf import create_surf_openai_client
from src.baseline.single_llm.schema import (
    PANGAEA_FULL_FIELDS,
    PANGAEA_FULL_STANDARD_NAME,
    PangaeaSingleLLMMetadata,
)
from src.config import get_surf_extra_body
from src.standards.standards import STANDARD_DEFINITIONS, get_schema_for_standard


DEFAULT_MODELS = [
    "default-text-large",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Sehyo/Qwen3.5-122B-A10B-NVFP4",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
]

CONTEXT_WINDOWS = {
    "default-text-large": 131_072,
    "Qwen/Qwen3.6-35B-A3B-FP8": 262_144,
    "Sehyo/Qwen3.5-122B-A10B-NVFP4": 262_144,
    "mistralai/Devstral-Small-2-24B-Instruct-2512": 262_144,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 131_072,
}

COMPLETION_RESERVE = 4_096
CHECKPOINT_VERSION = 3
BASELINE_DEFINITIONS = {
    **STANDARD_DEFINITIONS,
    PANGAEA_FULL_STANDARD_NAME: PANGAEA_FULL_FIELDS,
}


def _schema_for_standard(standard_name: str) -> Any:
    if standard_name == PANGAEA_FULL_STANDARD_NAME:
        return PangaeaSingleLLMMetadata
    return get_schema_for_standard(standard_name)

SYSTEM_MESSAGE = (
    "Find all metadata defined by the metadata standard. "
    "Use only the dataset text. "
    "Return one JSON object with exactly the standard's fields. "
    "Use null only for nullable fields that cannot be determined, and [] for "
    "lists with no supported values. Do not add fields."
)


def model_slug(model: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", model.replace("/", "_"))


@lru_cache(maxsize=1)
def _encoder() -> Any:
    return tiktoken.get_encoding("cl100k_base")


def prompt_tokens(system: str, user: str) -> int:
    """Approximate chat tokens with a safety margin before the SURF call.

    cl100k is not these models' tokenizer, so this cannot guarantee fit.
    """
    encoder = _encoder()
    raw = len(encoder.encode(system)) + len(encoder.encode(user)) + 32
    return int(raw * 1.15) + 1


def _json_shape(field_spec: dict[str, dict[str, Any]]) -> str:
    """Give the model valid JSON with the nested types expected by Pydantic."""
    shape: dict[str, Any] = {}
    for field_name, spec in field_spec.items():
        if field_name == "spatialCoverage":
            shape[field_name] = {
                "min_lat": 0.0, "min_lon": 0.0,
                "max_lat": 0.0, "max_lon": 0.0,
            }
        elif field_name == "temporalCoverage":
            shape[field_name] = {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}
        elif get_origin(spec["type"]) is list:
            shape[field_name] = []
        elif get_origin(spec["type"]) is dict:
            shape[field_name] = {}
        else:
            shape[field_name] = "value"
    return json.dumps(shape, indent=2)


def _coerce_lists(field_spec: dict[str, dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(payload)
    for field_name, spec in field_spec.items():
        value = coerced.get(field_name)
        if get_origin(spec["type"]) is list and isinstance(value, str):
            coerced[field_name] = [part.strip() for part in value.split(",") if part.strip()]
    return coerced


def _collapse_unknown_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn an all-null coverage object into null.

    The field itself is optional. A present object must contain real values,
    so ``{"min_lat": null, ...}`` is the model's way of saying unknown.
    """
    coerced = dict(payload)
    for field_name in ("spatialCoverage", "temporalCoverage"):
        value = coerced.get(field_name)
        if not isinstance(value, dict):
            continue
        present = {key: item for key, item in value.items() if item is not None}
        coerced[field_name] = present or None
    return coerced


def _validate_metadata(standard_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Regularize with the registered Pydantic model, then check useful coverage."""
    schema = _schema_for_standard(standard_name)
    if schema is None:
        raise KeyError(f"Unknown metadata standard: {standard_name}")
    expected = set(schema.model_fields)
    if set(payload) != expected:
        raise ValueError(
            f"Metadata fields differ from {standard_name}: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    field_spec = BASELINE_DEFINITIONS[standard_name]
    regularized = _collapse_unknown_coverage(_coerce_lists(field_spec, payload))
    metadata = schema.model_validate(regularized).model_dump(mode="json")
    if standard_name in {"croissant_pangaea_standard", PANGAEA_FULL_STANDARD_NAME}:
        if not isinstance(metadata["name"], str) or not metadata["name"].strip():
            raise ValueError("PANGAEA name must be nonempty")
        if not isinstance(metadata["description"], str) or not metadata["description"].strip():
            raise ValueError("PANGAEA description must be nonempty")
        if any(not keyword.strip() for keyword in metadata["keywords"]):
            raise ValueError("PANGAEA keywords must be nonempty strings")
    if standard_name in {"croissant_inaturalist_standard", PANGAEA_FULL_STANDARD_NAME}:
        spatial = metadata["spatialCoverage"]
        temporal = metadata["temporalCoverage"]
        if standard_name == "croissant_inaturalist_standard" and (
            spatial is None or temporal is None
        ):
            raise ValueError("iNaturalist requires spatial and temporal coverage")
        if spatial is not None:
            if set(spatial) != {"min_lat", "min_lon", "max_lat", "max_lon"}:
                raise ValueError("spatialCoverage needs four bounding-box values")
            if not all(math.isfinite(value) for value in spatial.values()):
                raise ValueError("coordinates must be finite")
            if not (-90 <= spatial["min_lat"] <= spatial["max_lat"] <= 90):
                raise ValueError("latitude bounds are invalid")
            if not (-180 <= spatial["min_lon"] <= spatial["max_lon"] <= 180):
                raise ValueError("longitude bounds are invalid")
        if temporal is not None:
            if set(temporal) != {"from", "to"}:
                raise ValueError("temporalCoverage needs from and to")
            try:
                first = datetime.fromisoformat(temporal["from"].replace("Z", "+00:00"))
                last = datetime.fromisoformat(temporal["to"].replace("Z", "+00:00"))
                if first > last:
                    raise ValueError("temporalCoverage is reversed")
            except (TypeError, ValueError) as exc:
                raise ValueError("temporalCoverage needs ordered ISO dates") from exc
        if standard_name == PANGAEA_FULL_STANDARD_NAME:
            for field_name in ("spatial", "temporal"):
                value = metadata[field_name]
                if value is not None and not value.strip():
                    raise ValueError(f"{field_name} resolution must be nonempty or null")
    return metadata


def _messages(standard_name: str, filename: str, dataset_text: str) -> tuple[str, str]:
    field_spec = BASELINE_DEFINITIONS[standard_name]
    definitions = []
    for field_name, spec in field_spec.items():
        hint = str(spec.get("prompt_hint") or "").strip()
        description = str(spec.get("description") or "").strip()
        detail = f"{field_name}: {description}"
        if hint:
            detail = f"{detail} {hint}"
        definitions.append(f"- {detail}")
    user = (
        f"Metadata standard: {standard_name}\n"
        "Field definitions:\n"
        + "\n".join(definitions)
        + "\n\nReturn JSON with this shape:\n"
        + _json_shape(field_spec)
        + "\nThe example values show types, not values to copy. "
          "For coverage, calculate bounds from all valid rows when the file "
          "contains numeric coordinates or dates. Do not infer coordinates "
          "from place names or invent unsupported resolution values."
        + f"\n\nDataset file: {filename}\n"
        + dataset_text
    )
    return SYSTEM_MESSAGE, user


def _checkpoint_identity(csv_path: Path, model_name: str, standard_name: str) -> dict[str, str]:
    """Invalidate old records when the source, prompt, or token policy changes."""
    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    system, prompt = _messages(standard_name, csv_path.name, "")
    contract = {
        "version": CHECKPOINT_VERSION,
        "system": system,
        "prompt": prompt,
        "context_window": CONTEXT_WINDOWS[model_name],
        "completion_reserve": COMPLETION_RESERVE,
        "token_estimator": "cl100k_base:1.15:32",
        "input_policy": "full_csv_utf8",
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"source_sha256": source_hash, "contract_sha256": contract_hash}


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response was not a JSON object")
    return payload


def _prepare_csv(
    csv_path: str | Path,
    *,
    model_name: str,
    standard_name: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Prepare the full-text prompt without creating a SURF client."""
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if model_name not in CONTEXT_WINDOWS:
        raise KeyError(f"No hardcoded context window for {model_name}")
    if _schema_for_standard(standard_name) is None:
        raise KeyError(f"Unknown metadata standard: {standard_name}")

    limit = CONTEXT_WINDOWS[model_name]
    base = {
        "csv": path.name,
        "dataset_id": path.stem,
        "method": "single_llm",
        "provider": "surf",
        "model": model_name,
        "standard": standard_name,
        "context_window": limit,
        "completion_reserve": COMPLETION_RESERVE,
        **_checkpoint_identity(path, model_name, standard_name),
    }
    try:
        dataset_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return (
            {**base, "status": "skipped", "reason": "unreadable", "message": str(exc)},
            None, None,
        )

    system, user = _messages(standard_name, path.name, dataset_text)
    tokens = prompt_tokens(system, user)
    base["prompt_tokens"] = tokens
    if tokens + COMPLETION_RESERVE > limit:
        return {**base, "status": "skipped", "reason": "context_window"}, None, None
    return base, system, user


def describe_csv(
    csv_path: str | Path,
    *,
    model_name: str,
    standard_name: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Return metadata for one CSV, or a skip record when it cannot fit."""
    base, system, user = _prepare_csv(
        csv_path, model_name=model_name, standard_name=standard_name
    )
    if base.get("status") == "skipped":
        return base
    assert system is not None and user is not None

    started = time.perf_counter()
    request: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": COMPLETION_RESERVE,
    }
    extra_body = get_surf_extra_body(model_name)
    if extra_body:
        request["extra_body"] = extra_body

    client = create_surf_openai_client(timeout=float(timeout_seconds))

    def complete(max_tokens: int) -> tuple[str | None, str]:
        response = client.chat.completions.create(**{**request, "max_tokens": max_tokens})
        if not response.choices:
            raise RuntimeError(f"SURF returned no choices for {model_name}")
        choice = response.choices[0]
        return choice.finish_reason, (choice.message.content or "").strip()

    def validated(text: str) -> dict[str, Any]:
        return _validate_metadata(standard_name, _parse_json_object(text))

    finish_reason, text = complete(COMPLETION_RESERVE)
    metadata: dict[str, Any] | None = None
    if text:
        try:
            metadata = validated(text)
        except (json.JSONDecodeError, ValidationError, ValueError):
            if finish_reason != "length":
                raise
    if metadata is None and finish_reason == "length":
        room = int(base["context_window"]) - int(base["prompt_tokens"])
        bumped = min(COMPLETION_RESERVE * 2, room)
        if bumped > COMPLETION_RESERVE:
            finish_reason, text = complete(bumped)
            if text:
                try:
                    metadata = validated(text)
                except (json.JSONDecodeError, ValidationError, ValueError):
                    if finish_reason != "length":
                        raise
    if metadata is None:
        if finish_reason == "length":
            raise RuntimeError(f"SURF truncated the single_llm response for {model_name}")
        raise RuntimeError(f"SURF returned an empty single_llm response for {model_name}")
    return {
        **base,
        "metadata": metadata,
        "finish_reason": finish_reason,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "status": "ok",
    }


def _describe_csv_worker(
    csv_path: str,
    kwargs: dict[str, Any],
    result_file: str,
    error_queue: multiprocessing.Queue,
) -> None:
    try:
        outcome = describe_csv(csv_path, **kwargs)
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(outcome, handle, ensure_ascii=False)
    except BaseException as exc:
        error_queue.put(f"{type(exc).__name__}: {exc}")


def describe_csv_with_timeout(
    csv_path: str | Path,
    *,
    model_name: str,
    standard_name: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one dataset in a killable child process."""
    ctx = multiprocessing.get_context("spawn")
    error_queue: multiprocessing.Queue = ctx.Queue()
    descriptor, result_file = tempfile.mkstemp(prefix="single_llm_", suffix=".json")
    os.close(descriptor)
    Path(result_file).unlink(missing_ok=True)
    kwargs = {
        "model_name": model_name,
        "standard_name": standard_name,
        "timeout_seconds": timeout_seconds,
    }
    process = ctx.Process(
        target=_describe_csv_worker,
        args=(str(csv_path), kwargs, result_file, error_queue),
    )
    process.start()
    process.join(timeout_seconds)
    try:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(2)
            raise TimeoutError(
                f"single_llm exceeded {timeout_seconds:.0f}s on {Path(csv_path).name}"
            )
        if not error_queue.empty():
            raise RuntimeError(error_queue.get())
        path = Path(result_file)
        if not path.exists():
            raise RuntimeError(
                f"single_llm worker exited without a result for {Path(csv_path).name}"
            )
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        Path(result_file).unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _success_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / f"{csv_path.stem}_single_llm.json"


def _skip_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / f"{csv_path.stem}_single_llm.skipped.json"


def _valid_success(
    path: Path, model: str, standard_name: str, identity: dict[str, str]
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if not (
        payload.get("status") == "ok"
        and payload.get("model") == model
        and payload.get("standard") == standard_name
        and all(payload.get(key) == value for key, value in identity.items())
        and isinstance(payload.get("metadata"), dict)
    ):
        return False
    try:
        _validate_metadata(standard_name, payload["metadata"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _valid_skip(
    path: Path, model: str, standard_name: str, identity: dict[str, str]
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(
        payload.get("status") == "skipped"
        and payload.get("reason") in {"context_window", "unreadable", "timeout"}
        and payload.get("model") == model
        and payload.get("standard") == standard_name
        and all(payload.get(key) == value for key, value in identity.items())
    )


def _processed(
    output_dir: Path, csv_path: Path, model: str, standard_name: str,
    identity: dict[str, str],
) -> bool:
    return _valid_success(
        _success_path(output_dir, csv_path), model, standard_name, identity
    ) or _valid_skip(_skip_path(output_dir, csv_path), model, standard_name, identity)


def model_status(
    output_dir: Path,
    csv_files: Iterable[Path],
    model: str,
    standard_name: str,
    identities: dict[Path, dict[str, str]] | None = None,
) -> dict[str, int]:
    files = list(csv_files)
    if identities is None:
        identities = {
            path: _checkpoint_identity(path, model, standard_name) for path in files
        }
    successful = sum(
        _valid_success(_success_path(output_dir, path), model, standard_name, identities[path])
        for path in files
    )
    skipped = sum(
        _valid_skip(_skip_path(output_dir, path), model, standard_name, identities[path])
        and not _valid_success(
            _success_path(output_dir, path), model, standard_name, identities[path]
        )
        for path in files
    )
    return {
        "expected": len(files),
        "successful": successful,
        "skipped": skipped,
        "processed": successful + skipped,
        "remaining": len(files) - successful - skipped,
    }


def run_model(
    *,
    model: str,
    csv_files: Iterable[Path],
    outputs_root: Path,
    standard_name: str,
    timeout_seconds: float = 600,
    experiment: str = "pangaea",
) -> dict[str, Any]:
    """Run or resume one model. Successes and context-window skips persist."""
    files = sorted(Path(path) for path in csv_files)
    output_dir = outputs_root / f"single_llm_{experiment}_{model_slug(model)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run_log.jsonl"
    identities = {
        path: _checkpoint_identity(path, model, standard_name) for path in files
    }
    before = model_status(output_dir, files, model, standard_name, identities)
    remaining = [
        path for path in files
        if not _processed(output_dir, path, model, standard_name, identities[path])
    ]
    print("\n" + "=" * 78)
    print(f"single_llm model: {model}")
    print(f"Standard: {standard_name}")
    print(f"Output: {output_dir}")
    if not remaining:
        _write_json_atomic(
            output_dir / "_model_complete.json",
            {"model": model, "output_dir": str(output_dir), **before},
        )
        print(
            "MODEL ALREADY COMPLETE — skipping "
            f"({before['successful']} success, {before['skipped']} skipped)."
        )
        return {"model": model, "output_dir": str(output_dir), **before}

    print(
        f"Resuming with {len(remaining)}/{len(files)} datasets remaining; "
        f"{before['processed']} already recorded."
    )
    for index, csv_path in enumerate(remaining, start=1):
        print(f"\n[{index}/{len(remaining)}] {csv_path.name}")
        started = time.perf_counter()
        try:
            preflight, _, _ = _prepare_csv(
                csv_path, model_name=model, standard_name=standard_name
            )
            outcome = (
                preflight if preflight.get("status") == "skipped"
                else describe_csv_with_timeout(
                    csv_path,
                    model_name=model,
                    standard_name=standard_name,
                    timeout_seconds=timeout_seconds,
                )
            )
            if outcome.get("status") == "skipped":
                skip_file = _skip_path(output_dir, csv_path)
                _write_json_atomic(skip_file, outcome)
                _append_log(run_log, {**outcome, "skip_file": str(skip_file)})
                print(
                    f"  SKIPPED {outcome.get('reason')} "
                    f"({outcome.get('prompt_tokens')} tokens, "
                    f"window {outcome.get('context_window')}) -> {skip_file.name}"
                )
                continue
            output_file = _success_path(output_dir, csv_path)
            _write_json_atomic(output_file, outcome)
            _append_log(run_log, {**outcome, "output_file": str(output_file)})
            print(f"  OK in {outcome.get('runtime_seconds')}s -> {output_file.name}")
        except TimeoutError as exc:
            record = {
                "status": "skipped",
                "reason": "timeout",
                "csv": csv_path.name,
                "dataset_id": csv_path.stem,
                "method": "single_llm",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                **identities[csv_path],
                "timeout_seconds": timeout_seconds,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "message": str(exc),
            }
            skip_file = _skip_path(output_dir, csv_path)
            _write_json_atomic(skip_file, record)
            _append_log(run_log, {**record, "skip_file": str(skip_file)})
            print(f"  TIMEOUT -> {skip_file.name}")
        except Exception as exc:
            event = {
                "status": "failed",
                "csv": csv_path.name,
                "dataset_id": csv_path.stem,
                "method": "single_llm",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _append_log(run_log, event)
            print(f"  FAILED (will retry after restart): {type(exc).__name__}: {exc}")
            traceback.print_exc()

    after = model_status(output_dir, files, model, standard_name, identities)
    summary: dict[str, Any] = {"model": model, "output_dir": str(output_dir), **after}
    if after["remaining"] == 0:
        _write_json_atomic(output_dir / "_model_complete.json", summary)
        print(
            f"\nMODEL COMPLETE: {after['successful']} success, "
            f"{after['skipped']} skipped. Moving to the next model."
        )
    else:
        (output_dir / "_model_complete.json").unlink(missing_ok=True)
        print(
            f"\nMODEL INCOMPLETE: {after['remaining']} retryable failure(s). "
            "The notebook will continue to the next model and retry these "
            "after a restart."
        )
    return summary


def run_all_models(
    *,
    data_dir: Path,
    outputs_root: Path,
    standard_name: str,
    models: Iterable[str] = DEFAULT_MODELS,
    timeout_seconds: float = 600,
    experiment: str = "pangaea",
    dataset_label: str = "PANGAEA",
) -> list[dict[str, Any]]:
    """Run all requested models sequentially with model/dataset-level resume."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if standard_name not in BASELINE_DEFINITIONS:
        raise KeyError(f"Unknown metadata standard: {standard_name}")

    selected_models = list(models)
    print(f"{dataset_label} datasets: {len(csv_files)}")
    print(f"Standard: {standard_name}")
    print(f"Models: {len(selected_models)}")
    summaries = []
    for number, model in enumerate(selected_models, start=1):
        print(f"\nMODEL {number}/{len(selected_models)}")
        summaries.append(
            run_model(
                model=model,
                csv_files=csv_files,
                outputs_root=outputs_root,
                standard_name=standard_name,
                timeout_seconds=timeout_seconds,
                experiment=experiment,
            )
        )
    return summaries
