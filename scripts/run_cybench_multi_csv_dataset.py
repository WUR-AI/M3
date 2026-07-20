#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def normalize_inaturalist_metadata(metadata: Any) -> Optional[Dict[str, Any]]:
    """Keep only the croissant-* subset shaped spatial/temporal fields."""
    if not isinstance(metadata, dict):
        return None
    if not {"spatialCoverage", "temporalCoverage"}.issubset(metadata.keys()):
        return None

    spatial = metadata.get("spatialCoverage")
    temporal = metadata.get("temporalCoverage")

    sp_keys = ("min_lat", "min_lon", "max_lat", "max_lon")
    tp_keys = ("from", "to")

    if spatial is not None:
        if not isinstance(spatial, dict) or not set(sp_keys).issubset(spatial.keys()):
            return None
        spatial = {key: spatial[key] for key in sp_keys}

    if temporal is not None:
        if not isinstance(temporal, dict) or not set(tp_keys).issubset(temporal.keys()):
            return None
        temporal = {key: temporal[key] for key in tp_keys}

    if spatial is None and temporal is None:
        return None

    return {"spatialCoverage": spatial, "temporalCoverage": temporal}


def extract_metadata(metadata_result: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction to match the notebook behavior."""
    if metadata_result is None:
        return None

    candidates: List[Any] = []

    if getattr(metadata_result, "final_metadata", None):
        candidates.append(metadata_result.final_metadata)

    workspace = getattr(metadata_result, "final_workspace", None) or {}
    if workspace.get("metadata_output") is not None:
        candidates.append(workspace.get("metadata_output"))

    for candidate in candidates:
        normalized = normalize_inaturalist_metadata(candidate)
        if normalized:
            return normalized

    # Fallback: scan step results for a plausible JSON/dict.
    for step in reversed(getattr(metadata_result, "step_results", None) or []):
        if getattr(step, "player_role", None) != "croissant_inaturalist_metadata_generator":
            continue
        for player_result in getattr(step, "individual_results", None) or []:
            analysis = player_result.get("analysis")
            if not analysis:
                continue

            # If it is a dict already, normalize directly.
            if isinstance(analysis, dict):
                normalized = normalize_inaturalist_metadata(analysis)
                if normalized:
                    return normalized

            # If it is a string containing JSON, try to extract the outer {...}.
            if isinstance(analysis, str) and analysis:
                try:
                    start = analysis.find("{")
                    end = analysis.rfind("}")
                    if start != -1 and end > start:
                        parsed = json.loads(analysis[start : end + 1])
                        normalized = normalize_inaturalist_metadata(parsed)
                        if normalized:
                            return normalized
                except json.JSONDecodeError:
                    pass

    return None


def is_empty_metadata(metadata: Optional[Dict[str, Any]]) -> bool:
    return metadata is None or not metadata.get("spatialCoverage") and not metadata.get("temporalCoverage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cybench folder as one multi_csv dataset.")
    parser.add_argument("--input-dir", required=True, help="Folder containing the cybench *.csv files.")
    parser.add_argument("--dataset-tag", default=None, help="Tag for output subfolder (defaults to input-dir basename).")
    parser.add_argument(
        "--context-prefix",
        default=None,
        help="Prefix used for context name and output filename (defaults to cybench_<dataset-tag>).",
    )
    parser.add_argument("--csv-pattern", default="*.csv", help="Glob pattern for CSV files inside input-dir.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if output json exists.")

    parser.add_argument("--llm-provider", default="surf", help="LLM provider override.")
    parser.add_argument(
        "--llm-model",
        default="Sehyo/Qwen3.5-122B-A10B-NVFP4",
        help="LLM model override.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    load_dotenv(REPO_ROOT / ".env")
    os.environ["LLM_PROVIDER"] = args.llm_provider
    os.environ["LLM_MODEL"] = args.llm_model

    from src.config import get_model_name
    from src.orchestrator import run_metadata_extraction
    from src.standards import METADATA_STANDARDS

    STANDARD_NAME = "croissant_standard_subset"
    TOPOLOGY = "single"

    input_dir = Path(args.input_dir).resolve()
    dataset_tag = args.dataset_tag or input_dir.name
    context_prefix = args.context_prefix or f"cybench_{dataset_tag}"

    csv_paths = sorted(input_dir.glob(args.csv_pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching {args.csv_pattern} in {input_dir}")

    metadata_standard = METADATA_STANDARDS[STANDARD_NAME]
    llm_name = get_model_name()
    llm_slug = re.sub(r"[^\\w.\\-]+", "_", llm_name.replace("/", "_"))
    experiment_run = f"cybench_{llm_slug}"

    output_dir = REPO_ROOT / "outputs" / experiment_run / dataset_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{context_prefix}_multi_csv.json"

    print(f"LLM provider: {os.environ.get('LLM_PROVIDER')}")
    print(f"LLM model: {llm_name}")
    print(f"Experiment run: {experiment_run}")
    print(f"Input dir: {input_dir}")
    print(f"Found CSV files: {len(csv_paths)}")
    print(f"Output: {output_file}")

    if args.skip_if_exists and output_file.exists():
        print("Skipping (output already exists).")
        return

    # One call: pass a list of CSV paths so CSVContext becomes MULTI_CSV.
    result = None
    for attempt in (1, 2):
        try:
            print(f"Attempt {attempt}: running multi_csv extraction...")
            result = run_metadata_extraction(
                source=[str(p.resolve()) for p in csv_paths],
                metadata_standard=metadata_standard,
                metadata_standard_name=STANDARD_NAME,
                name=context_prefix,
                topology_name=TOPOLOGY,
            )
            break
        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc}")
            result = None

    metadata = extract_metadata(result)
    metadata = normalize_inaturalist_metadata(metadata)

    if is_empty_metadata(metadata):
        raise RuntimeError(f"Multi-csv extraction produced empty metadata for: {input_dir}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Saved JSON.")


if __name__ == "__main__":
    main()

