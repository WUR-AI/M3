from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline.autoddg.autoddg import AutoDDG
from src.baseline.autoddg.utils.sampling import get_sample
from src.config import LLM_PROVIDER, get_model_name


def describe_csv(
    csv_path: str | Path,
    *,
    sample_size: int = 20,
    description_words: int = 100,
    random_state: int = 9,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run the AutoDDG profile/topic/description pipeline on one CSV."""
    path = Path(csv_path)
    started = time.perf_counter()
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if df.empty:
        raise ValueError(f"{path.name} is empty")

    resolved_model = get_model_name(model_name)
    pipe = AutoDDG.from_surf(model_name=resolved_model, description_words=description_words)
    sample_df, sample_csv = get_sample(df, sample_size, random_state=random_state)
    dataset_profile, datamart_semantic_notes = pipe.profile_dataframe(sample_df)
    semantic_profile = pipe.analyze_semantics(
        sample_df,
        use_group_prompting=True,
        group_size=0,
    )
    title = path.stem
    topic = pipe.generate_topic(title, None, sample_csv)
    prompt, description = pipe.describe_dataset(
        dataset_sample=sample_csv,
        dataset_profile=dataset_profile,
        use_profile=True,
        semantic_profile=semantic_profile,
        use_semantic_profile=True,
        data_topic=topic,
        use_topic=True,
    )
    return {
        "csv": path.name,
        "dataset_id": path.stem,
        "title": title,
        "provider": LLM_PROVIDER,
        "model": resolved_model,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "sample_size": int(len(sample_df)),
        "topic": topic,
        "dataset_profile": dataset_profile,
        "datamart_semantic_notes": datamart_semantic_notes,
        "semantic_profile": semantic_profile,
        "prompt": prompt,
        "description": description,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "status": "ok",
    }


def _describe_csv_worker(
    csv_path: str,
    kwargs: dict[str, Any],
    result_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
) -> None:
    try:
        result_queue.put(describe_csv(csv_path, **kwargs))
    except BaseException as exc:
        error_queue.put(f"{type(exc).__name__}: {exc}")


def describe_csv_with_timeout(
    csv_path: str | Path,
    *,
    timeout_seconds: float = 600,
    sample_size: int = 20,
    description_words: int = 100,
    random_state: int = 9,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run ``describe_csv`` in a child process and kill it if it exceeds the timeout."""
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    error_queue: multiprocessing.Queue = ctx.Queue()
    kwargs = {
        "sample_size": sample_size,
        "description_words": description_words,
        "random_state": random_state,
        "model_name": model_name,
    }
    process = ctx.Process(
        target=_describe_csv_worker,
        args=(str(csv_path), kwargs, result_queue, error_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(2)
        raise TimeoutError(
            f"AutoDDG exceeded {timeout_seconds:.0f}s on {Path(csv_path).name}"
        )
    if not error_queue.empty():
        raise RuntimeError(error_queue.get())
    if result_queue.empty():
        raise RuntimeError(
            f"AutoDDG worker exited without a result for {Path(csv_path).name}"
        )
    return result_queue.get()
