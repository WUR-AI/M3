from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from src.baseline.dataatlas.csv_runner import describe_csv_with_timeout


DEFAULT_MODELS = [
    "default-text-large",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Sehyo/Qwen3.5-122B-A10B-NVFP4",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
]


def model_slug(model: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", model.replace("/", "_"))


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
    return output_dir / f"{csv_path.stem}_dataatlas.json"


def _skip_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / f"{csv_path.stem}_dataatlas.skipped.json"


def _raw_path(output_dir: Path, csv_path: Path) -> Path:
    return output_dir / "raw" / csv_path.stem / "dictionary.json"


def _valid_success(path: Path, model: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "ok"
        and payload.get("model") == model
        and str(payload.get("description") or "").strip()
    )


def _valid_timeout(path: Path, model: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("status") == "skipped"
        and payload.get("reason") == "timeout"
        and payload.get("model") == model
    )


def _processed(output_dir: Path, csv_path: Path, model: str) -> bool:
    return _valid_success(_success_path(output_dir, csv_path), model) or _valid_timeout(
        _skip_path(output_dir, csv_path), model
    )


def model_status(
    output_dir: Path, csv_files: Iterable[Path], model: str
) -> dict[str, int]:
    files = list(csv_files)
    successful = sum(
        _valid_success(_success_path(output_dir, path), model) for path in files
    )
    timed_out = sum(
        _valid_timeout(_skip_path(output_dir, path), model) for path in files
    )
    return {
        "expected": len(files),
        "successful": successful,
        "timed_out": timed_out,
        "processed": successful + timed_out,
        "remaining": len(files) - successful - timed_out,
    }


def run_model(
    *,
    model: str,
    csv_files: Iterable[Path],
    outputs_root: Path,
    timeout_seconds: float = 600,
    domain: str = "Earth and environmental science",
    max_rows: int | None = 20_000_000,
    experiment: str = "pangaea",
) -> dict[str, Any]:
    """Run or resume one model; successes and timeouts are persistent."""
    files = sorted(Path(path) for path in csv_files)
    output_dir = outputs_root / f"dataatlas_{experiment}_{model_slug(model)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run_log.jsonl"

    before = model_status(output_dir, files, model)
    remaining = [path for path in files if not _processed(output_dir, path, model)]
    print("\n" + "=" * 78)
    print(f"DataAtlas model: {model}")
    print(f"Output: {output_dir}")
    if not remaining:
        print(
            "MODEL ALREADY COMPLETE — skipping "
            f"({before['successful']} success, {before['timed_out']} timeout)."
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
            bundle = describe_csv_with_timeout(
                csv_path,
                model_name=model,
                timeout_seconds=timeout_seconds,
                domain=domain,
                max_rows=max_rows,
            )
            result = bundle["result"]
            raw_file = _raw_path(output_dir, csv_path)
            _write_json_atomic(raw_file, bundle["dictionary"])
            result["raw_dictionary"] = str(raw_file)
            output_file = _success_path(output_dir, csv_path)
            _write_json_atomic(output_file, result)
            _append_log(
                run_log,
                {
                    **result,
                    "output_file": str(output_file),
                    "raw_dictionary": str(raw_file),
                },
            )
            print(
                f"  OK in {result['runtime_seconds']}s -> {output_file.name}\n"
                f"  {result['description']}"
            )
        except TimeoutError as exc:
            record = {
                "status": "skipped",
                "reason": "timeout",
                "csv": csv_path.name,
                "dataset_id": csv_path.stem,
                "method": "DataAtlas",
                "provider": "surf",
                "model": model,
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
                "method": "DataAtlas",
                "provider": "surf",
                "model": model,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            _append_log(run_log, event)
            print(f"  FAILED (will retry after restart): {type(exc).__name__}: {exc}")
            traceback.print_exc()

    after = model_status(output_dir, files, model)
    summary: dict[str, Any] = {"model": model, "output_dir": str(output_dir), **after}
    if after["remaining"] == 0:
        _write_json_atomic(output_dir / "_model_complete.json", summary)
        print(
            f"\nMODEL COMPLETE: {after['successful']} success, "
            f"{after['timed_out']} timeout. Moving to the next model."
        )
    else:
        print(
            f"\nMODEL INCOMPLETE: {after['remaining']} retryable failure(s). "
            "The notebook will continue to the next model now and retry these "
            "after a restart."
        )
    return summary


def run_all_models(
    *,
    data_dir: Path,
    outputs_root: Path,
    models: Iterable[str] = DEFAULT_MODELS,
    timeout_seconds: float = 600,
    domain: str = "Earth and environmental science",
    max_rows: int | None = 20_000_000,
    experiment: str = "pangaea",
    dataset_label: str = "PANGAEA",
) -> list[dict[str, Any]]:
    """Run all requested models sequentially with model/dataset-level resume."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    selected_models = list(models)
    print(f"{dataset_label} datasets: {len(csv_files)}")
    print(f"Models: {len(selected_models)}")
    summaries = []
    for number, model in enumerate(selected_models, start=1):
        print(f"\nMODEL {number}/{len(selected_models)}")
        summaries.append(
            run_model(
                model=model,
                csv_files=csv_files,
                outputs_root=outputs_root,
                timeout_seconds=timeout_seconds,
                domain=domain,
                max_rows=max_rows,
                experiment=experiment,
            )
        )
    return summaries
