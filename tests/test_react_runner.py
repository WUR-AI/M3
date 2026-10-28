"""Offline checks for the ReAct baseline loop."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from src.baseline.ReAct import runner
from src.baseline.single_llm.schema import PANGAEA_FULL_STANDARD_NAME


class _Structured:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, messages: list) -> dict:
        assert messages
        return self.payload


class _Scripted:
    def __init__(self, replies: list[AIMessage]) -> None:
        self.replies = replies
        self.invocations = 0

    def bind_tools(self, tools: list):
        assert tools
        return self

    def invoke(self, messages: list) -> AIMessage:
        self.invocations += 1
        return self.replies.pop(0)

    def with_structured_output(self, schema, **kwargs):
        return _Structured({
            "name": "Dataset",
            "description": "One value",
            "keywords": ["value"],
            "spatialCoverage": None,
            "temporalCoverage": None,
            "spatial": None,
            "temporal": None,
        })


def test_tool_loop_injects_context_and_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    scripted = _Scripted([
        AIMessage(content="", tool_calls=[{
            "name": "get_field_names", "args": {}, "id": "call-1", "type": "tool_call",
        }]),
        AIMessage(content="The only column is value."),
    ])
    monkeypatch.setattr(runner, "create_llm", lambda **kwargs: scripted)
    outcome = runner.describe_csv(
        source, model_name="default-text-large", standard_name=PANGAEA_FULL_STANDARD_NAME
    )
    assert outcome["status"] == "ok"
    assert outcome["metadata"]["name"] == "Dataset"
    assert outcome["tool_calls"][0]["tool"] == "get_field_names"
    assert outcome["tool_calls"][0]["args"]["context_key"] == "dataset"
    assert scripted.invocations == 2


def test_multi_csv_registers_one_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "yield.csv"
    second = tmp_path / "meteo.csv"
    first.write_text("value\n1\n", encoding="utf-8")
    second.write_text("value\n2\n", encoding="utf-8")
    scripted = _Scripted([
        AIMessage(content="", tool_calls=[{
            "name": "get_field_names", "args": {}, "id": "call-1", "type": "tool_call",
        }]),
        AIMessage(content="Two tables."),
    ])
    monkeypatch.setattr(runner, "create_llm", lambda **kwargs: scripted)
    outcome = runner.describe_multi_csv(
        [first, second],
        model_name="default-text-large",
        standard_name=PANGAEA_FULL_STANDARD_NAME,
        dataset_id="cybench_IE",
    )
    assert outcome["status"] == "ok"
    assert outcome["dataset_id"] == "cybench_IE"
    assert outcome["csvs"] == ["yield.csv", "meteo.csv"]
    assert outcome["tool_calls"][0]["args"]["context_key"] == "cybench_IE"


def test_existing_result_file_is_not_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    source = source_dir / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    outputs = tmp_path / "outputs"

    def _refuse(*args, **kwargs):
        raise AssertionError("existing result was sent again")

    monkeypatch.setattr(runner, "describe_csv_with_timeout", _refuse)
    output_dir = outputs / "react_pangaea_default-text-large"
    output_dir.mkdir(parents=True)
    (output_dir / "dataset_react.json").write_text('{"status": "ok"}', encoding="utf-8")
    summary = runner.run_model(
        model="default-text-large",
        csv_files=[source],
        outputs_root=outputs,
        standard_name=PANGAEA_FULL_STANDARD_NAME,
        experiment="pangaea",
    )
    assert summary["output_dir"].endswith("react_pangaea_default-text-large")
    assert (output_dir / "dataset_react.json").read_text(encoding="utf-8") == '{"status": "ok"}'


def test_oversize_turn_is_not_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "dataset.csv"
    source.write_text("value\n1\n", encoding="utf-8")

    class _Blocked(_Scripted):
        def invoke(self, messages: list) -> AIMessage:
            raise AssertionError("oversize turn was sent")

    monkeypatch.setattr(runner, "create_llm", lambda **kwargs: _Blocked([]))
    monkeypatch.setattr(runner, "prompt_tokens", lambda *args, **kwargs: 10**9)
    outcome = runner.describe_csv(
        source, model_name="default-text-large", standard_name=PANGAEA_FULL_STANDARD_NAME
    )
    assert outcome["status"] == "skipped"
    assert outcome["reason"] == "context_window"
