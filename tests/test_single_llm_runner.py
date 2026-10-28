"""Offline checks for the one-call baseline's output and resume contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baseline.single_llm import runner
from src.baseline.single_llm.schema import PANGAEA_FULL_STANDARD_NAME
from src.standards.standards import STANDARD_DEFINITIONS


def test_prompt_shape_uses_nested_coverage_objects() -> None:
    shape = json.loads(
        runner._json_shape(STANDARD_DEFINITIONS["croissant_inaturalist_standard"])
    )
    assert set(shape["spatialCoverage"]) == {
        "min_lat", "min_lon", "max_lat", "max_lon"
    }
    assert set(shape["temporalCoverage"]) == {"from", "to"}
    pangaea_shape = json.loads(
        runner._json_shape(runner.BASELINE_DEFINITIONS[PANGAEA_FULL_STANDARD_NAME])
    )
    assert set(pangaea_shape) == {
        "name", "description", "keywords", "spatialCoverage",
        "temporalCoverage", "spatial", "temporal",
    }


def test_pydantic_regularizes_keywords_but_rejects_empty_metadata() -> None:
    metadata = runner._validate_metadata(
        "croissant_pangaea_standard",
        {"name": "Dataset", "description": "Measurements", "keywords": "water, sediment"},
    )
    assert metadata["keywords"] == ["water", "sediment"]

    with pytest.raises(ValueError, match="description must be nonempty"):
        runner._validate_metadata(
            "croissant_pangaea_standard",
            {"name": "Dataset", "description": None, "keywords": []},
        )
    with pytest.raises(ValueError, match="missing"):
        runner._validate_metadata(
            "croissant_inaturalist_standard", {"spatialCoverage": None}
        )


def test_inaturalist_requires_complete_ordered_coverage() -> None:
    payload = {
        "spatialCoverage": {
            "min_lat": -35, "min_lon": 150,
            "max_lat": -34, "max_lon": 151,
        },
        "temporalCoverage": {"from": "2020-01-01", "to": "2024-12-31"},
    }
    assert runner._validate_metadata("croissant_inaturalist_standard", payload) == payload
    payload["spatialCoverage"]["max_lat"] = -36
    with pytest.raises(ValueError, match="latitude bounds"):
        runner._validate_metadata("croissant_inaturalist_standard", payload)


def test_pangaea_combines_descriptive_and_nullable_technical_fields() -> None:
    payload = {
        "name": "Dataset", "description": "Measurements", "keywords": ["water"],
        "spatialCoverage": None,
        "temporalCoverage": {"from": "2020-01-01", "to": "2021-12-31"},
        "spatial": None, "temporal": "monthly",
    }
    result = runner._validate_metadata(PANGAEA_FULL_STANDARD_NAME, payload)
    assert result["description"] == "Measurements"
    assert result["temporalCoverage"] == payload["temporalCoverage"]
    assert result["spatialCoverage"] is None

    payload["spatialCoverage"] = {"min_lat": -35}
    with pytest.raises(ValueError, match="bounding-box"):
        runner._validate_metadata(PANGAEA_FULL_STANDARD_NAME, payload)

    payload["spatialCoverage"] = {
        "min_lat": None, "min_lon": None, "max_lat": None, "max_lon": None,
    }
    payload["temporalCoverage"] = {"from": None, "to": None}
    result = runner._validate_metadata(PANGAEA_FULL_STANDARD_NAME, payload)
    assert result["spatialCoverage"] is None
    assert result["temporalCoverage"] is None


def test_resume_rechecks_source_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    model = "default-text-large"
    standard = "croissant_pangaea_standard"
    identity = runner._checkpoint_identity(source, model, standard)
    success = tmp_path / "dataset_single_llm.json"
    record = {
        "status": "ok", "model": model, "standard": standard,
        "metadata": {"name": "Dataset", "description": "Values", "keywords": []},
        **identity,
    }
    success.write_text(json.dumps(record), encoding="utf-8")
    assert runner._valid_success(success, model, standard, identity)

    record["metadata"]["description"] = None
    success.write_text(json.dumps(record), encoding="utf-8")
    assert not runner._valid_success(success, model, standard, identity)

    source.write_text("value\n2\n", encoding="utf-8")
    assert not runner._valid_success(
        success, model, standard, runner._checkpoint_identity(source, model, standard)
    )


def test_resume_rechecks_prompt_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    before = runner._checkpoint_identity(
        source, "default-text-large", "croissant_pangaea_standard"
    )
    monkeypatch.setattr(runner, "SYSTEM_MESSAGE", runner.SYSTEM_MESSAGE + " New instruction.")
    after = runner._checkpoint_identity(
        source, "default-text-large", "croissant_pangaea_standard"
    )
    assert before["source_sha256"] == after["source_sha256"]
    assert before["contract_sha256"] != after["contract_sha256"]


def test_full_csv_is_sent_once_when_it_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n2\n", encoding="utf-8")
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps({
                    "name": "Dataset", "description": "Two values", "keywords": ["values"],
                    "spatialCoverage": None, "temporalCoverage": None,
                    "spatial": None, "temporal": None,
                })),
            )]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(runner, "create_surf_openai_client", lambda **kwargs: client)
    outcome = runner.describe_csv(
        source, model_name="default-text-large", standard_name=PANGAEA_FULL_STANDARD_NAME
    )
    assert outcome["status"] == "ok"
    assert set(outcome["metadata"]) == set(runner.BASELINE_DEFINITIONS[PANGAEA_FULL_STANDARD_NAME])
    assert len(requests) == 1
    assert requests[0]["messages"][1]["content"].endswith("value\n1\n2\n")


def test_truncated_reply_is_retried_once_with_a_larger_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    budgets: list[int] = []
    replies = [
        ("length", "{"),
        ("stop", json.dumps({
            "name": "Dataset", "description": "One value", "keywords": ["value"],
            "spatialCoverage": {
                "min_lat": None, "min_lon": None, "max_lat": None, "max_lon": None,
            },
            "temporalCoverage": {"from": None, "to": None},
            "spatial": None, "temporal": None,
        })),
    ]

    def create(**kwargs):
        budgets.append(kwargs["max_tokens"])
        finish_reason, content = replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(runner, "create_surf_openai_client", lambda **kwargs: client)
    outcome = runner.describe_csv(
        source, model_name="default-text-large", standard_name=PANGAEA_FULL_STANDARD_NAME
    )
    assert outcome["status"] == "ok"
    assert outcome["metadata"]["spatialCoverage"] is None
    assert budgets == [runner.COMPLETION_RESERVE, runner.COMPLETION_RESERVE * 2]


def test_oversize_csv_is_skipped_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setitem(runner.CONTEXT_WINDOWS, "default-text-large", 100)
    monkeypatch.setattr(
        runner, "create_surf_openai_client",
        lambda **kwargs: pytest.fail("Client must not be created for a skip"),
    )
    outcome = runner.describe_csv(
        source, model_name="default-text-large", standard_name="croissant_pangaea_standard"
    )
    assert outcome["status"] == "skipped"
    assert outcome["reason"] == "context_window"


def test_run_model_records_skip_without_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setitem(runner.CONTEXT_WINDOWS, "default-text-large", 100)
    monkeypatch.setattr(
        runner, "describe_csv_with_timeout",
        lambda *args, **kwargs: pytest.fail("Skip must not start a worker"),
    )
    args = {
        "model": "default-text-large",
        "csv_files": [source],
        "outputs_root": tmp_path / "outputs",
        "standard_name": "croissant_pangaea_standard",
    }
    first = runner.run_model(**args)
    assert first["skipped"] == 1
    assert first["remaining"] == 0
    skip_path = Path(first["output_dir"]) / "dataset_single_llm.skipped.json"
    original = json.loads(skip_path.read_text(encoding="utf-8"))
    assert runner.run_model(**args)["skipped"] == 1

    source.write_text("value\n2\n", encoding="utf-8")
    rerun = runner.run_model(**args)
    assert rerun["skipped"] == 1
    replacement = json.loads(skip_path.read_text(encoding="utf-8"))
    assert replacement["source_sha256"] != original["source_sha256"]
