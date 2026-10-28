from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline.dataatlas.surf import run_surf_llm


def _normalize_mixed_timezone_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Make datetime columns comparable before DataAtlas aggregates them.

    iNaturalist files mix timezone-aware and timezone-naive values. ``eventDate``
    stores ``2021-06-05T08:25Z`` beside ``2022-06-11T11:31``, while the CSV
    reader types ``lastInterpreted`` as UTC and ``dateIdentified`` as naive.
    DataAtlas takes ``min``/``max`` across those values, and Pandas rejects the
    mix with ``can't compare offset-naive and offset-aware datetimes``. The
    whole table is then dropped. Zoned instants are converted to UTC and stored
    without an offset so every datetime column compares as naive. Columns that
    are already entirely naive are left unchanged.
    """
    for column in dataframe.select_dtypes(include="datetimetz").columns:
        dataframe[column] = (
            dataframe[column].dt.tz_convert("UTC").dt.tz_localize(None)
        )

    zoned_pattern = r"(?:Z|[+-]\d{2}:?\d{2})$"
    dated_pattern = r"\d{4}-\d{2}-\d{2}"
    for column in dataframe.select_dtypes(include="object").columns:
        text = dataframe[column].dropna().astype(str)
        if text.empty:
            continue
        zoned = text.str.contains(zoned_pattern, regex=True)
        dated = text.str.contains(dated_pattern, regex=True)
        if not (zoned.any() and (dated & ~zoned).any()):
            continue
        parsed = pd.to_datetime(
            text.loc[zoned], errors="coerce", utc=True, format="mixed"
        ).dropna()
        if parsed.empty:
            continue
        dataframe.loc[parsed.index, column] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return dataframe


def _configure_dataatlas_compatibility() -> Any:
    """Configure the pinned upstream package for SURF and Pandas 3.

    DataAtlas downcasts low-cardinality strings to unordered categoricals.
    Pandas 3 preserves that categorical dtype through ``to_datetime`` and then
    rejects min/max date aggregation. Converting categories back to ordinary
    objects preserves their values and restores the behavior expected by the
    upstream profiler.

    Mixed timezone columns are normalized to naive UTC values before
    profiling so date min/max does not crash and drop the table.
    """
    import data_dictionary_generator.generate_dictionary as generator
    import data_dictionary_generator.load_dataset as loader
    import data_dictionary_generator.metadata_extraction as extraction

    original_read_dataset = loader.read_dataset

    def read_dataset_compatible(*args: Any, **kwargs: Any) -> Any:
        dataframe = original_read_dataset(*args, **kwargs)
        categorical_columns = list(dataframe.select_dtypes(include="category").columns)
        if categorical_columns:
            dataframe[categorical_columns] = dataframe[categorical_columns].astype(
                "object"
            )
        return _normalize_mixed_timezone_columns(dataframe)

    extraction.read_dataset = read_dataset_compatible
    generator.run_llm_dispatcher = run_surf_llm
    return generator


def describe_csv(
    csv_path: str | Path,
    *,
    model_name: str,
    domain: str = "Earth and environmental science",
    max_rows: int | None = 20_000_000,
) -> dict[str, Any]:
    """Run the published DataAtlas pipeline on exactly one CSV.

    The upstream package is monkey-patched only at its dispatcher boundary so
    its metadata extraction, prompts, and 15-word post-processing stay intact.
    """
    try:
        import data_dictionary_generator
    except ImportError as exc:
        raise ImportError(
            "DataAtlas is not installed. Install the pinned clone with: "
            "python -m pip install -e "
            "/home/com3dian/Github/process_data_metadata_agent/"
            "data-dictionary-generator --no-deps"
        ) from exc

    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"DataAtlas input must be a CSV: {path.name}")

    started = time.perf_counter()
    generator = _configure_dataatlas_compatibility()
    metadata = generator.generate_full_metadata_and_descriptions(
        dataset_path=path,
        output_dir=path.parent,
        llm_model=model_name,
        domain_covered=domain,
        max_workers=1,
        max_rows=max_rows,
    )

    table = metadata.tables.get(path.stem)
    if table is None:
        available = ", ".join(metadata.tables) or "none"
        raise RuntimeError(
            f"DataAtlas did not return table {path.stem!r}; available: {available}"
        )
    if not table.description:
        raise RuntimeError(f"DataAtlas returned no table description for {path.name}")

    raw_dictionary = metadata.model_dump(mode="json", exclude_none=True)
    result = {
        "csv": path.name,
        "dataset_id": path.stem,
        "method": "DataAtlas",
        "dataatlas_version": getattr(data_dictionary_generator, "__version__", None),
        "provider": "surf",
        "model": model_name,
        "domain": domain,
        "n_rows": int(table.row_count),
        "n_columns": int(table.number_of_columns),
        "description": table.description,
        "dataset_description": metadata.description,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "status": "ok",
    }
    return {"result": result, "dictionary": raw_dictionary}


def _describe_csv_worker(
    csv_path: str,
    kwargs: dict[str, Any],
    result_file: str,
    error_queue: multiprocessing.Queue,
) -> None:
    try:
        bundle = describe_csv(csv_path, **kwargs)
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False)
    except BaseException as exc:
        error_queue.put(f"{type(exc).__name__}: {exc}")


def describe_csv_with_timeout(
    csv_path: str | Path,
    *,
    model_name: str,
    timeout_seconds: float = 600,
    domain: str = "Earth and environmental science",
    max_rows: int | None = 20_000_000,
) -> dict[str, Any]:
    """Run one DataAtlas dataset in a killable child process."""
    ctx = multiprocessing.get_context("spawn")
    error_queue: multiprocessing.Queue = ctx.Queue()
    descriptor, result_file = tempfile.mkstemp(prefix="dataatlas_", suffix=".json")
    os.close(descriptor)
    Path(result_file).unlink(missing_ok=True)

    kwargs = {
        "model_name": model_name,
        "domain": domain,
        "max_rows": max_rows,
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
                f"DataAtlas exceeded {timeout_seconds:.0f}s on {Path(csv_path).name}"
            )
        if not error_queue.empty():
            raise RuntimeError(error_queue.get())
        path = Path(result_file)
        if not path.exists():
            raise RuntimeError(
                f"DataAtlas worker exited without a result for {Path(csv_path).name}"
            )
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        Path(result_file).unlink(missing_ok=True)
