"""One ReAct loop per dataset, using the agent's CSV tools and context.

The CSV is registered as an ExecutionContext. The model inspects it through
``get_single_csv_tools``. When it stops calling tools, the next step writes
the same structured metadata as the single-prompt baseline. A turn that does
not fit the hardcoded context window is not sent.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from src.baseline.single_llm.runner import (
    BASELINE_DEFINITIONS,
    COMPLETION_RESERVE,
    CONTEXT_WINDOWS,
    DEFAULT_MODELS,
    _json_shape,
    _parse_json_object,
    _schema_for_standard,
    _validate_metadata,
    model_slug,
    prompt_tokens,
)
from src.baseline.single_llm.schema import PANGAEA_FULL_STANDARD_NAME
from src.config import PLAYER_MAX_TOOL_ITERATIONS, create_llm
from src.context import create_context
from src.tools.context_tools import (
    clear_registry,
    get_multi_csv_tools,
    get_single_csv_tools,
    register_context,
)


MAX_STEPS = PLAYER_MAX_TOOL_ITERATIONS*2

SYSTEM_MESSAGE = (
    "Find all metadata defined by the metadata standard. "
    "Use the tools to inspect the registered dataset. "
    "The execution environment fills context_key when you omit it. "
    "Do not invent coordinates or dates from place names or sample values. "
    "When the tool results are enough, stop calling tools."
)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _dump(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def _message_tokens(messages: list[Any]) -> int:
    parts = []
    for message in messages:
        parts.append(type(message).__name__)
        parts.append(_text(getattr(message, "content", "")))
        calls = getattr(message, "tool_calls", None)
        if calls:
            parts.append(_dump(calls))
    return prompt_tokens("", "\n".join(parts))


def _fits(model_name: str, messages: list[Any]) -> tuple[bool, int]:
    tokens = _message_tokens(messages)
    return tokens + COMPLETION_RESERVE <= CONTEXT_WINDOWS[model_name], tokens


def _task_message(standard_name: str, filename: str, context_key: str) -> str:
    field_spec = BASELINE_DEFINITIONS[standard_name]
    definitions = []
    for field_name, spec in field_spec.items():
        hint = str(spec.get("prompt_hint") or "").strip()
        description = str(spec.get("description") or "").strip()
        detail = f"{field_name}: {description}"
        if hint:
            detail = f"{detail} {hint}"
        definitions.append(f"- {detail}")
    return (
        f"Metadata standard: {standard_name}\n"
        "Field definitions:\n"
        + "\n".join(definitions)
        + "\n\nFill this shape from tool results:\n"
        + _json_shape(field_spec)
        + "\nThe example values show types, not values to copy. "
        "Use null for an unknown nullable field and [] for an empty list.\n"
        f"Dataset file: {filename}\n"
        f"context_key: {context_key}"
    )


def _checkpoint_identity(
    csv_path: Path, model_name: str, standard_name: str
) -> dict[str, str]:
    tool_names = [tool.name for tool in get_single_csv_tools()]
    contract = {
        "system": SYSTEM_MESSAGE,
        "tools": tool_names,
        "max_steps": MAX_STEPS,
        "standard": standard_name,
        "shape": _json_shape(BASELINE_DEFINITIONS[standard_name]),
        "context_window": CONTEXT_WINDOWS[model_name],
        "completion_reserve": COMPLETION_RESERVE,
        "input_policy": "execution_context_tools",
    }
    return {
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _tool_args(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("args")
    if raw is None and isinstance(call.get("function"), dict):
        raw = call["function"].get("arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return dict(raw) if isinstance(raw, dict) else {}


def _run_tool(tool: Any, args: dict[str, Any], context_key: str) -> Any:
    merged = dict(args)
    merged["context_key"] = context_key
    if tool.args_schema is not None:
        merged = tool.args_schema.model_validate(merged).model_dump()
    return tool.invoke(merged)


def _structured_metadata(llm: Any, schema: Any, standard_name: str, messages: list[Any]) -> dict[str, Any]:
    errors: list[Exception] = []
    for method in (None, "function_calling"):
        try:
            kwargs = {} if method is None else {"method": method}
            result = llm.with_structured_output(schema, **kwargs).invoke(messages)
            if hasattr(result, "model_dump"):
                payload = result.model_dump(mode="json")
            elif isinstance(result, dict):
                payload = result
            else:
                raise ValueError("Structured output was not an object")
            return _validate_metadata(standard_name, payload)
        except Exception as exc:
            errors.append(exc)
    text = _text(llm.invoke(messages + [HumanMessage(
        content="Return one JSON object and nothing else."
    )]).content)
    try:
        return _validate_metadata(standard_name, _parse_json_object(text))
    except Exception as exc:
        errors.append(exc)
        raise RuntimeError(
            "ReAct structured output failed: " + "; ".join(f"{type(err).__name__}: {err}" for err in errors)
        ) from exc


def _execute_react(
    *,
    base: dict[str, Any],
    schema: Any,
    standard_name: str,
    model_name: str,
    context_key: str,
    tools: list[Any],
    task_text: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Tool loop for an already registered dataset. The next step after a stop is structured output."""
    tools_by_name = {tool.name: tool for tool in tools}
    llm = create_llm(
        model_name=model_name,
        temperature=0,
        request_timeout=float(timeout_seconds),
    )
    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as exc:
        raise RuntimeError(f"Tool calling is unavailable for {model_name}: {exc}") from exc

    messages: list[Any] = [
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=task_text),
    ]
    tool_calls: list[dict[str, Any]] = []
    started = time.perf_counter()
    metadata: dict[str, Any] | None = None
    for step in range(MAX_STEPS):
        fits, tokens = _fits(model_name, messages)
        base["prompt_tokens"] = tokens
        if not fits:
            return {**base, "status": "skipped", "reason": "context_window", "step": step}
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        calls = list(getattr(ai_msg, "tool_calls", None) or [])
        if not calls:
            messages.append(HumanMessage(content=(
                "Return the required metadata from the tool results."
            )))
            fits, tokens = _fits(model_name, messages)
            base["prompt_tokens"] = tokens
            if not fits:
                return {**base, "status": "skipped", "reason": "context_window", "step": step}
            metadata = _structured_metadata(llm, schema, standard_name, messages)
            break
        for index, call in enumerate(calls):
            name = call.get("name")
            if not name and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            args = _tool_args(call)
            call_id = call.get("id") or f"step{step}_{index}"
            tool = tools_by_name.get(name or "")
            if tool is None:
                out: Any = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    out = _run_tool(tool, args, context_key)
                except (ValidationError, Exception) as exc:
                    out = {"error": str(exc)}
            tool_calls.append({"tool": name, "args": {**args, "context_key": context_key}})
            messages.append(ToolMessage(content=_dump(out), tool_call_id=call_id))
    if metadata is None:
        messages.append(HumanMessage(content=(
            "Stop calling tools. Return the required metadata from the tool results."
        )))
        fits, tokens = _fits(model_name, messages)
        base["prompt_tokens"] = tokens
        if not fits:
            return {**base, "status": "skipped", "reason": "context_window", "step": MAX_STEPS}
        metadata = _structured_metadata(llm, schema, standard_name, messages)
    return {
        **base,
        "metadata": metadata,
        "tool_calls": tool_calls,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "status": "ok",
    }


def describe_csv(
    csv_path: str | Path,
    *,
    model_name: str,
    standard_name: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one single-CSV ReAct dataset, or return a skip record when a turn cannot fit."""
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if model_name not in CONTEXT_WINDOWS:
        raise KeyError(f"No hardcoded context window for {model_name}")
    schema = _schema_for_standard(standard_name)
    if schema is None:
        raise KeyError(f"Unknown metadata standard: {standard_name}")

    identity = _checkpoint_identity(path, model_name, standard_name)
    clear_registry()
    context_key = register_context(path.stem, create_context(str(path), name=path.stem))
    return _execute_react(
        base={
            "csv": path.name,
            "dataset_id": path.stem,
            "method": "react",
            "provider": "surf",
            "model": model_name,
            "standard": standard_name,
            "context_window": CONTEXT_WINDOWS[model_name],
            "completion_reserve": COMPLETION_RESERVE,
            **identity,
        },
        schema=schema,
        standard_name=standard_name,
        model_name=model_name,
        context_key=context_key,
        tools=get_single_csv_tools(),
        task_text=_task_message(standard_name, path.name, context_key),
        timeout_seconds=timeout_seconds,
    )


def _multi_checkpoint_identity(
    paths: list[Path], model_name: str, standard_name: str
) -> dict[str, str]:
    tool_names = [tool.name for tool in get_multi_csv_tools()]
    contract = {
        "system": SYSTEM_MESSAGE,
        "tools": tool_names,
        "max_steps": MAX_STEPS,
        "standard": standard_name,
        "shape": _json_shape(BASELINE_DEFINITIONS[standard_name]),
        "context_window": CONTEXT_WINDOWS[model_name],
        "completion_reserve": COMPLETION_RESERVE,
        "input_policy": "multi_csv_execution_context_tools",
    }
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "source_sha256": digest.hexdigest(),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def describe_multi_csv(
    csv_paths: Iterable[str | Path],
    *,
    model_name: str,
    standard_name: str,
    dataset_id: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one multi-CSV folder as a single ReAct dataset."""
    paths = [Path(path).resolve() for path in csv_paths]
    if not paths:
        raise FileNotFoundError("No CSV files were provided")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if model_name not in CONTEXT_WINDOWS:
        raise KeyError(f"No hardcoded context window for {model_name}")
    schema = _schema_for_standard(standard_name)
    if schema is None:
        raise KeyError(f"Unknown metadata standard: {standard_name}")

    identity = _multi_checkpoint_identity(paths, model_name, standard_name)
    clear_registry()
    context_key = register_context(
        dataset_id,
        create_context([str(path) for path in paths], name=dataset_id),
    )
    filenames = ", ".join(path.name for path in paths)
    return _execute_react(
        base={
            "csvs": [path.name for path in paths],
            "dataset_id": dataset_id,
            "method": "react",
            "provider": "surf",
            "model": model_name,
            "standard": standard_name,
            "context_window": CONTEXT_WINDOWS[model_name],
            "completion_reserve": COMPLETION_RESERVE,
            **identity,
        },
        schema=schema,
        standard_name=standard_name,
        model_name=model_name,
        context_key=context_key,
        tools=get_multi_csv_tools(),
        task_text=_task_message(standard_name, filenames, context_key),
        timeout_seconds=timeout_seconds,
    )


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
    descriptor, result_file = tempfile.mkstemp(prefix="react_", suffix=".json")
    os.close(descriptor)
    Path(result_file).unlink(missing_ok=True)
    process = ctx.Process(
        target=_describe_csv_worker,
        args=(
            str(csv_path),
            {
                "model_name": model_name,
                "standard_name": standard_name,
                "timeout_seconds": timeout_seconds,
            },
            result_file,
            error_queue,
        ),
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
                f"react exceeded {timeout_seconds:.0f}s on {Path(csv_path).name}"
            )
        if not error_queue.empty():
            raise RuntimeError(error_queue.get())
        path = Path(result_file)
        if not path.exists():
            raise RuntimeError(
                f"react worker exited without a result for {Path(csv_path).name}"
            )
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        Path(result_file).unlink(missing_ok=True)


def _describe_multi_csv_worker(
    csv_paths: list[str],
    kwargs: dict[str, Any],
    result_file: str,
    error_queue: multiprocessing.Queue,
) -> None:
    try:
        outcome = describe_multi_csv(csv_paths, **kwargs)
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(outcome, handle, ensure_ascii=False)
    except BaseException as exc:
        error_queue.put(f"{type(exc).__name__}: {exc}")


def describe_multi_csv_with_timeout(
    csv_paths: Iterable[str | Path],
    *,
    model_name: str,
    standard_name: str,
    dataset_id: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Run one multi-CSV dataset in a killable child process."""
    paths = [str(Path(path).resolve()) for path in csv_paths]
    ctx = multiprocessing.get_context("spawn")
    error_queue: multiprocessing.Queue = ctx.Queue()
    descriptor, result_file = tempfile.mkstemp(prefix="react_", suffix=".json")
    os.close(descriptor)
    Path(result_file).unlink(missing_ok=True)
    process = ctx.Process(
        target=_describe_multi_csv_worker,
        args=(
            paths,
            {
                "model_name": model_name,
                "standard_name": standard_name,
                "dataset_id": dataset_id,
                "timeout_seconds": timeout_seconds,
            },
            result_file,
            error_queue,
        ),
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
                f"react exceeded {timeout_seconds:.0f}s on {dataset_id}"
            )
        if not error_queue.empty():
            raise RuntimeError(error_queue.get())
        path = Path(result_file)
        if not path.exists():
            raise RuntimeError(f"react worker exited without a result for {dataset_id}")
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
    return output_dir / f"{csv_path.stem}_react.json"


def _skip_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / f"{csv_path.stem}_react.skipped.json"


def _has_saved_result(output_dir: Path, csv_path: Path) -> bool:
    """True when a previous run already wrote a success or skip file."""
    return _success_path(output_dir, csv_path).is_file() or _skip_path(
        output_dir, csv_path
    ).is_file()


def _write_result(path: Path, payload: Any) -> bool:
    """Write a result file. An existing file is left unchanged."""
    if path.is_file():
        return False
    _write_json_atomic(path, payload)
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _matches(payload: dict[str, Any], model: str, standard_name: str, identity: dict[str, str]) -> bool:
    return bool(
        payload.get("model") == model
        and payload.get("standard") == standard_name
        and all(payload.get(key) == value for key, value in identity.items())
    )


def _valid_success(path: Path, model: str, standard_name: str, identity: dict[str, str]) -> bool:
    payload = _read_json(path)
    if payload is None or payload.get("status") != "ok" or not _matches(payload, model, standard_name, identity):
        return False
    if not isinstance(payload.get("metadata"), dict):
        return False
    try:
        _validate_metadata(standard_name, payload["metadata"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _valid_skip(path: Path, model: str, standard_name: str, identity: dict[str, str]) -> bool:
    payload = _read_json(path)
    return bool(
        payload
        and payload.get("status") == "skipped"
        and payload.get("reason") in {"context_window", "timeout", "failed"}
        and _matches(payload, model, standard_name, identity)
    )


def _processed(
    output_dir: Path, csv_path: Path, model: str, standard_name: str, identity: dict[str, str]
) -> bool:
    return _valid_success(
        _success_path(output_dir, csv_path), model, standard_name, identity
    ) or _valid_skip(_skip_path(output_dir, csv_path), model, standard_name, identity)


def model_status(
    output_dir: Path,
    csv_files: Iterable[Path],
    model: str,
    standard_name: str,
    identities: dict[Path, dict[str, str]],
) -> dict[str, int]:
    files = list(csv_files)
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
    """Run or resume one model. An existing result file is never replaced."""
    files = sorted(Path(path) for path in csv_files)
    output_dir = outputs_root / f"react_{experiment}_{model_slug(model)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run_log.jsonl"
    identities = {
        path: _checkpoint_identity(path, model, standard_name) for path in files
    }
    before = model_status(output_dir, files, model, standard_name, identities)
    remaining = [path for path in files if not _has_saved_result(output_dir, path)]
    print("\n" + "=" * 78)
    print(f"react model: {model}")
    print(f"Standard: {standard_name}")
    print(f"Output: {output_dir}")
    if not remaining:
        if before["remaining"] == 0:
            _write_json_atomic(
                output_dir / "_model_complete.json",
                {"model": model, "output_dir": str(output_dir), **before},
            )
            print(
                "MODEL ALREADY COMPLETE — skipping "
                f"({before['successful']} success, {before['skipped']} skipped)."
            )
        else:
            print(
                "EXISTING RESULTS KEPT — "
                f"{len(files)} files already present, not overwritten."
            )
        return {"model": model, "output_dir": str(output_dir), **before}

    print(
        f"Resuming with {len(remaining)}/{len(files)} datasets remaining; "
        f"{len(files) - len(remaining)} existing result files kept."
    )
    for index, csv_path in enumerate(remaining, start=1):
        print(f"\n[{index}/{len(remaining)}] {csv_path.name}")
        if _has_saved_result(output_dir, csv_path):
            print("  KEPT existing result, not overwritten")
            continue
        started = time.perf_counter()
        try:
            outcome = describe_csv_with_timeout(
                csv_path,
                model_name=model,
                standard_name=standard_name,
                timeout_seconds=timeout_seconds,
            )
            if outcome.get("status") == "skipped":
                skip_file = _skip_path(output_dir, csv_path)
                if not _write_result(skip_file, outcome):
                    print(f"  KEPT existing result, not overwritten -> {skip_file.name}")
                    continue
                _append_log(run_log, {**outcome, "skip_file": str(skip_file)})
                print(
                    f"  SKIPPED {outcome.get('reason')} "
                    f"({outcome.get('prompt_tokens')} tokens, "
                    f"window {outcome.get('context_window')}) -> {skip_file.name}"
                )
                continue
            output_file = _success_path(output_dir, csv_path)
            if not _write_result(output_file, outcome):
                print(f"  KEPT existing result, not overwritten -> {output_file.name}")
                continue
            _append_log(run_log, {**outcome, "output_file": str(output_file)})
            called = ", ".join(call["tool"] for call in outcome.get("tool_calls") or []) or "none"
            print(f"  OK in {outcome.get('runtime_seconds')}s tools: {called} -> {output_file.name}")
        except TimeoutError as exc:
            record = {
                "status": "skipped",
                "reason": "timeout",
                "csv": csv_path.name,
                "dataset_id": csv_path.stem,
                "method": "react",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                "timeout_seconds": timeout_seconds,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "message": str(exc),
                **identities[csv_path],
            }
            skip_file = _skip_path(output_dir, csv_path)
            if not _write_result(skip_file, record):
                print(f"  KEPT existing result, not overwritten -> {skip_file.name}")
                continue
            _append_log(run_log, {**record, "skip_file": str(skip_file)})
            print(f"  TIMEOUT -> {skip_file.name}")
        except Exception as exc:
            record = {
                "status": "skipped",
                "reason": "failed",
                "csv": csv_path.name,
                "dataset_id": csv_path.stem,
                "method": "react",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "message": str(exc),
                **identities[csv_path],
            }
            skip_file = _skip_path(output_dir, csv_path)
            if not _write_result(skip_file, record):
                print(f"  KEPT existing result, not overwritten -> {skip_file.name}")
                continue
            _append_log(run_log, {**record, "skip_file": str(skip_file)})
            print(f"  FAILED once, not retried -> {skip_file.name}: {type(exc).__name__}: {exc}")
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


def run_cybench(
    *,
    data_dir: Path,
    outputs_root: Path,
    standard_name: str = "croissant_standard_subset",
    models: Iterable[str] = DEFAULT_MODELS,
    timeout_seconds: float = 600,
    experiment: str = "cybench",
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run one Cybench folder as a single multi-CSV dataset for each model.

    An existing result file is kept and not replaced.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if standard_name not in BASELINE_DEFINITIONS:
        raise KeyError(f"Unknown metadata standard: {standard_name}")

    dataset_id = dataset_id or f"cybench_{data_dir.name}"
    label = Path(f"{dataset_id}.csv")
    selected_models = list(models)
    print(f"Cybench directory: {data_dir}")
    print(f"CSV files: {len(csv_files)}")
    print(f"Dataset: {dataset_id}")
    print(f"Standard: {standard_name}")
    print(f"Models: {len(selected_models)}")
    summaries = []
    for number, model in enumerate(selected_models, start=1):
        print(f"\nMODEL {number}/{len(selected_models)}")
        print("\n" + "=" * 78)
        print(f"react model: {model}")
        output_dir = outputs_root / f"react_{experiment}_{model_slug(model)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")
        if _has_saved_result(output_dir, label):
            print("EXISTING RESULT KEPT — not overwritten.")
            summaries.append({
                "model": model,
                "output_dir": str(output_dir),
                "dataset_id": dataset_id,
                "status": "kept",
            })
            continue
        started = time.perf_counter()
        try:
            outcome = describe_multi_csv_with_timeout(
                csv_files,
                model_name=model,
                standard_name=standard_name,
                dataset_id=dataset_id,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as exc:
            outcome = {
                "status": "skipped",
                "reason": "timeout",
                "dataset_id": dataset_id,
                "method": "react",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                "timeout_seconds": timeout_seconds,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "message": str(exc),
            }
        except Exception as exc:
            outcome = {
                "status": "skipped",
                "reason": "failed",
                "dataset_id": dataset_id,
                "method": "react",
                "provider": "surf",
                "model": model,
                "standard": standard_name,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            traceback.print_exc()
        if outcome.get("status") == "skipped":
            destination = _skip_path(output_dir, label)
        else:
            destination = _success_path(output_dir, label)
        if not _write_result(destination, outcome):
            print(f"  KEPT existing result, not overwritten -> {destination.name}")
        else:
            called = ", ".join(call["tool"] for call in outcome.get("tool_calls") or []) or "none"
            if outcome.get("status") == "ok":
                print(f"  OK in {outcome.get('runtime_seconds')}s tools: {called} -> {destination.name}")
            else:
                print(f"  {outcome.get('reason')} -> {destination.name}")
        summaries.append({
            "model": model,
            "output_dir": str(output_dir),
            "dataset_id": dataset_id,
            "status": outcome.get("status"),
            "result_file": destination.name,
        })
    return summaries


__all__ = [
    "CONTEXT_WINDOWS",
    "DEFAULT_MODELS",
    "PANGAEA_FULL_STANDARD_NAME",
    "describe_csv",
    "describe_multi_csv",
    "run_all_models",
    "run_cybench",
    "run_model",
]
