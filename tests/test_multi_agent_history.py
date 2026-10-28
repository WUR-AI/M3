"""Regression tests for evidence continuity across tool and debate rounds."""

import json
from threading import Barrier

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.run_trace import RunTraceRecorder
from src.core.schemas import ExecutionResult, Plan, Task
from src.context.base_context import ContextType
from src.orchestrator.plan_executor import PlanExecutor
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.utils import apply_verified_coverage, verified_tool_observations
from src.orchestrator.step_executor import get_step_execution_graph
from src.players.player import Player


@tool
def observed_value(context_key: str) -> dict:
    """Return a source-derived value for the test."""
    return {"value": 7, "context": context_key}


class ScriptedModel:
    def __init__(self):
        self.inputs = []
        self.responses = [
            AIMessage(content="", tool_calls=[{"name": "observed_value", "args": {}, "id": "first"}]),
            AIMessage(content="Initial finding: 7"),
            AIMessage(content="Peer should verify the same source"),
            AIMessage(content="", tool_calls=[{"name": "observed_value", "args": {}, "id": "second"}]),
            AIMessage(content="Revised finding: 7"),
            AIMessage(content="Final finding: 7"),
        ]

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.inputs.append(list(messages))
        return self.responses.pop(0)


def test_revision_continues_tool_conversation_and_saves_io(tmp_path):
    model = ScriptedModel()
    player = object.__new__(Player)
    player.name = "analyst_1"
    player.role_prompt = "Check the data."
    player.tools = [observed_value]
    player.llm = model
    player.max_tool_iterations = 4
    player.role_key = "analyst"
    player.require_tool_calls = True
    player.required_tools = ["observed_value"]
    recorder = RunTraceRecorder(tmp_path / "run.jsonl")

    first = player.execute_task(
        task="Read the value", context_key="dataset", context_info={"resources": ["data"]},
        workspace={}, inputs={}, recorder=recorder, step_index=0,
    )
    critique = player.critique_work(
        task="Read the value", other_players_work={"analyst_2": "Found 7"},
        messages=first["messages"], recorder=recorder, step_index=0,
    )
    revised = player.revise_work(
        task="Read the value", messages=critique["messages"],
        critiques=["Check the source again"], context_key="dataset",
        tool_results=first["tool_results"], tool_trace=first["tool_trace"],
        recorder=recorder, step_index=0,
    )
    synthesized = player.synthesize_results(
        task="Read the value", all_results=[
            {"player": "analyst_1", "analysis": revised["analysis"],
             "tool_results": revised["tool_results"], "conversation": revised["messages"]},
            {"player": "analyst_2", "analysis": "Found 7", "tool_results": {},
             "conversation": [AIMessage(content="Peer original answer: 7")]},
        ],
        conversation=revised["messages"], recorder=recorder, step_index=0,
    )

    assert first["analysis"] == "Initial finding: 7"
    assert critique["analysis"] == "Peer should verify the same source"
    assert revised["analysis"] == "Revised finding: 7"
    assert synthesized == "Final finding: 7"
    assert len(revised["tool_results"]) == 2
    assert any(isinstance(message, ToolMessage) for message in model.inputs[3])
    assert "Check the source again" in model.inputs[3][-1].content
    assert any(isinstance(message, ToolMessage) for message in model.inputs[5])
    assert "Found 7" in model.inputs[5][-1].content
    assert "Peer original answer: 7" not in model.inputs[5][-1].content
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [event["event"] for event in events].count("model_input") == 6
    assert [event["event"] for event in events].count("tool_output") == 2
    assert events[0]["messages"][0]["content"].startswith("You are analyst_1")


class DebatePlayer:
    def __init__(self, name, barrier):
        self.name = name
        self.barrier = barrier
        self.synthesis_inputs = None

    def execute_task(self, **kwargs):
        self.barrier.wait(timeout=2)
        return {
            "analysis": f"{self.name} found 7", "tool_results": {"source": {"value": 7}},
            "tool_trace": [{"tool": "observed_value", "result": {"value": 7}}],
            "messages": [f"{self.name} original conversation"], "success": True,
        }

    def critique_work(self, **kwargs):
        assert "Verified tool results" in next(iter(kwargs["other_players_work"].values()))
        assert kwargs["messages"][0] == f"{self.name} original conversation" or "revised" in kwargs["messages"]
        return {"analysis": "Check the source", "messages": kwargs["messages"] + ["critique"]}

    def revise_work(self, **kwargs):
        assert kwargs["messages"][0] == f"{self.name} original conversation"
        assert kwargs["tool_results"]["source"] == {"value": 7}
        return {
            "analysis": f"{self.name} verified 7", "tool_results": kwargs["tool_results"],
            "tool_trace": kwargs["tool_trace"],
            "messages": kwargs["messages"] + ["revised"], "success": True,
        }

    def synthesize_results(self, **kwargs):
        self.synthesis_inputs = kwargs["all_results"]
        return "verified 7"


def test_multi_agent_step_runs_concurrently_and_keeps_evidence(tmp_path):
    barrier = Barrier(2)
    players = [DebatePlayer("analyst_1", barrier), DebatePlayer("analyst_2", barrier)]
    state = {
        "step_index": 0, "task": "Check data", "player_name": "analyst",
        "rationale": "test", "input_mappings": {}, "expected_outputs": ["facts"],
        "target_resources": [], "context_key": "dataset",
        "context_info": {"resources": ["data"], "is_multi_csv": False},
        "workspace": {}, "metadata_standard": "{}", "tool_tracer": None,
        "recorder": RunTraceRecorder(tmp_path / "debate.jsonl"),
        "players": players, "synthesizer": players[0],
        "max_debate_rounds": 2, "current_debate_round": 0,
        "player_results": [], "agent_messages": {}, "debate_log": [],
        "output_schema": None, "consolidated_result": None,
        "produced_artifacts": {}, "error": None,
    }
    result = get_step_execution_graph().invoke(state)

    assert result["error"] is None
    assert result["current_debate_round"] == 2
    assert result["produced_artifacts"] == {"facts": "verified 7"}
    assert all(r["tool_results"]["source"] == {"value": 7} for r in result["player_results"])
    assert all(len(messages) == 5 for messages in result["agent_messages"].values())
    assert all(candidate["tool_results"]["source"] == {"value": 7} for candidate in players[0].synthesis_inputs)
    assert all("conversation" not in candidate for candidate in players[0].synthesis_inputs)


def test_error_tool_result_does_not_satisfy_required_call():
    player = object.__new__(Player)
    player.require_tool_calls = True
    player.required_tools = ["observed_value"]
    failed = [{"source": "llm", "tool": "observed_value", "args": {"resource": "a"},
               "result": {"error": "source unavailable"}}]
    assert player._missing_required_tools(failed, ["a"]) == ["observed_value(resource='a')"]
    failed[0]["result"] = '{"error": "source unavailable"}'
    assert player._missing_required_tools(failed, ["a"]) == ["observed_value(resource='a')"]


class MinimalContext:
    name = "sample"
    description = ""
    resources = ["data"]
    context_type = ContextType.SINGLE_CSV
    is_multi_csv = False

    def to_dict(self):
        return {"name": self.name, "resources": self.resources}

    def get_schema(self):
        return {"data": ["value"]}

    def get_relationships(self):
        return []


def test_executor_rejects_empty_final_metadata(monkeypatch, tmp_path):
    plan = Plan(steps=[
        Task(task="Inspect data", player="data_analyst", rationale="source",
             outputs=["facts"]),
        Task(task="Generate metadata", player="metadata_generator", rationale="finish",
             inputs={"facts": "facts", "metadata_standard": "metadata_standard"},
             outputs=["metadata_output"]),
    ])

    class EmptyGraph:
        def invoke(self, state):
            if self.calls == 0:
                self.calls += 1
                return {"produced_artifacts": {"facts": "observed"},
                        "consolidated_result": "observed", "player_results": [],
                        "current_debate_round": 0, "debate_log": []}
            return {"produced_artifacts": {"metadata_output": {"name": ""}},
                    "consolidated_result": {"name": ""}, "player_results": [],
                    "current_debate_round": 0, "debate_log": []}

        calls = 0

    monkeypatch.setattr("src.orchestrator.plan_executor.create_step_state", lambda **kwargs: {})
    executor = PlanExecutor("single")
    executor.step_graph = EmptyGraph()
    result = executor.execute(
        plan, MinimalContext(), "dataset", "{}",
        recorder=RunTraceRecorder(tmp_path / "empty.jsonl"),
    )
    assert not result.success
    assert result.final_metadata is None
    assert "nonempty final metadata" in result.error
    assert result.trace_path == str(tmp_path / "empty.jsonl")


def test_executor_stops_after_failed_dependency(monkeypatch, tmp_path):
    plan = Plan(steps=[
        Task(task="Inspect data", player="data_analyst", rationale="source",
             outputs=["facts"]),
        Task(task="Generate metadata", player="metadata_generator", rationale="finish",
             inputs={"facts": "facts", "metadata_standard": "metadata_standard"},
             outputs=["metadata_output"]),
    ])

    class FailedGraph:
        calls = 0

        def invoke(self, state):
            self.calls += 1
            return {"error": "tool failed", "player_results": [],
                    "current_debate_round": 0, "debate_log": []}

    monkeypatch.setattr("src.orchestrator.plan_executor.create_step_state", lambda **kwargs: {})
    executor = PlanExecutor("single")
    executor.step_graph = FailedGraph()
    result = executor.execute(
        plan, MinimalContext(), "dataset", "{}",
        recorder=RunTraceRecorder(tmp_path / "failed.jsonl"),
    )
    assert executor.step_graph.calls == 1
    assert not result.success
    assert len(result.step_results) == 1
    assert result.step_results[0].error == "tool failed"


def test_verified_observations_correct_event_date_and_bounds():
    raw = [{
        "tool_trace": [
            {"tool": "get_temporal_extent", "args": {"time_column": "eventDate"},
             "result": {"from": "2000-01-01T00:00:00+00:00", "to": "2020-06-10T10:00:00+00:00"}},
            {"tool": "get_temporal_extent", "args": {"time_column": "lastInterpreted"},
             "result": {"from": "2026-06-23", "to": "2026-06-23"}},
            {"tool": "get_spatial_extent", "args": {"lat_column": "decimalLatitude",
                                                  "lon_column": "decimalLongitude"},
             "result": {"bounding_box": {"min_lat": 10.0, "max_lat": 20.0,
                                         "min_lon": 30.0, "max_lon": 40.0}}},
            {"tool": "get_spatial_extent", "args": {"lat_column": "bad",
                                                  "lon_column": "bad"},
             "result": {"error": "no coordinates"}},
        ]
    }]
    evidence = verified_tool_observations(raw, 0)
    assert len(evidence) == 3
    output = {"spatialCoverage": None,
              "temporalCoverage": {"from": "2000-01-01", "to": "2026-06-23"}}
    fixed, changes = apply_verified_coverage(output, evidence, "croissant_inaturalist_standard")
    assert fixed["temporalCoverage"]["to"] == "2020-06-10T10:00:00+00:00"
    assert fixed["spatialCoverage"]["max_lon"] == 40.0
    assert {change["field"] for change in changes} == {"spatialCoverage", "temporalCoverage"}


def test_final_agent_receives_prior_verified_tool_results(tmp_path):
    class FinalModel:
        messages = None

        def invoke(self, messages):
            self.messages = list(messages)
            return AIMessage(content='{"name": "sample"}')

    model = FinalModel()
    player = object.__new__(Player)
    player.name = "metadata_generator_1"
    player.role_prompt = "Write metadata."
    player.tools = []
    player.llm = model
    player.role_key = "metadata_generator"
    result = player.execute_task(
        task="Generate metadata", context_key="dataset",
        context_info={"resources": ["data"]},
        workspace={"facts": "brief summary", "_verified_evidence": [
            {"tool": "get_temporal_extent", "args": {"time_column": "eventDate"},
             "result": {"to": "2020-06-10"}},
        ]},
        inputs={"facts": "facts"},
        recorder=RunTraceRecorder(tmp_path / "final.jsonl"),
    )
    assert result["success"]
    assert "Verified tool observations" in model.messages[1].content
    assert "2020-06-10" in model.messages[1].content


def test_executor_passes_tool_evidence_to_final_output(monkeypatch, tmp_path):
    plan = Plan(steps=[
        Task(task="Measure extents", player="spatial_temporal_specialist",
             rationale="source", outputs=["spatial_temporal_data"]),
        Task(task="Generate metadata", player="croissant_inaturalist_metadata_generator",
             rationale="finish", inputs={"spatial_temporal": "spatial_temporal_data",
                                         "metadata_standard": "metadata_standard"},
             outputs=["metadata_output"]),
    ])

    class CoverageGraph:
        calls = 0

        def invoke(self, state):
            self.calls += 1
            if self.calls == 1:
                return {
                    "produced_artifacts": {"spatial_temporal_data": "ambiguous summary"},
                    "consolidated_result": "ambiguous summary", "current_debate_round": 0,
                    "debate_log": [], "player_results": [{"tool_trace": [
                        {"tool": "get_temporal_extent", "args": {"time_column": "eventDate"},
                         "result": {"from": "2000-01-01", "to": "2020-06-10"}},
                    ]}],
                }
            return {
                "produced_artifacts": {"metadata_output": {
                    "spatialCoverage": None,
                    "temporalCoverage": {"from": "2000-01-01", "to": "2026-06-23"},
                }},
                "consolidated_result": "wrong date", "current_debate_round": 0,
                "debate_log": [], "player_results": [],
            }

    captured_states = []

    def capture_state(**kwargs):
        captured_states.append(kwargs)
        return {}

    monkeypatch.setattr("src.orchestrator.plan_executor.create_step_state", capture_state)
    executor = PlanExecutor("single")
    executor.step_graph = CoverageGraph()
    result = executor.execute(
        plan, MinimalContext(), "dataset", "{}",
        metadata_standard_name="croissant_inaturalist_standard",
        recorder=RunTraceRecorder(tmp_path / "coverage.jsonl"),
    )
    assert result.success
    assert result.final_metadata["temporalCoverage"]["to"] == "2020-06-10"
    assert captured_states[1]["workspace"]["_verified_evidence"][0]["tool"] == "get_temporal_extent"
    events = [json.loads(line) for line in (tmp_path / "coverage.jsonl").read_text().splitlines()]
    assert any(event["event"] == "metadata_correction" for event in events)


def test_orchestrator_records_planning_io_and_exposes_trace(monkeypatch, tmp_path):
    plan = Plan(steps=[
        Task(task="Inspect data", player="data_analyst", rationale="source",
             outputs=["facts"]),
        Task(task="Generate metadata", player="metadata_generator", rationale="finish",
             inputs={"facts": "facts", "metadata_standard": "metadata_standard"},
             outputs=["metadata_output"]),
    ])

    class PlannerModel:
        def invoke(self, messages):
            return AIMessage(content=plan.model_dump_json())

    trace_path = tmp_path / "public-run.jsonl"
    monkeypatch.setattr("src.orchestrator.orchestrator.create_llm", lambda **kwargs: PlannerModel())
    monkeypatch.setattr("src.orchestrator.orchestrator.create_context", lambda *args, **kwargs: MinimalContext())
    monkeypatch.setattr(
        RunTraceRecorder, "create",
        classmethod(lambda cls, name: cls(trace_path)),
    )
    orchestrator = Orchestrator(topology_name="single")
    monkeypatch.setattr(orchestrator, "_generate_player_manifest", lambda context: "data_analyst, metadata_generator")
    monkeypatch.setattr(
        orchestrator, "execute_plan",
        lambda **kwargs: ExecutionResult(
            plan_steps_count=2, steps_completed=2, success=True,
            final_metadata={"name": "sample"}, trace_path=str(kwargs["recorder"].path),
        ),
    )

    result = orchestrator.run(source="unused.csv", metadata_standard="name")
    assert result.success
    assert result.trace_path == orchestrator.last_trace_path == str(trace_path)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "run_start", "model_input", "model_output", "plan", "plan_validation", "run_end",
    ]
    assert events[1]["messages"][0]["content"]
    assert events[2]["message"]["content"] == plan.model_dump_json()

    class InvalidPlannerModel:
        def invoke(self, messages):
            return AIMessage(content='{"steps": []}')

    failed_trace = tmp_path / "failed-plan.jsonl"
    monkeypatch.setattr(
        RunTraceRecorder, "create",
        classmethod(lambda cls, name: cls(failed_trace)),
    )
    orchestrator.llm = InvalidPlannerModel()
    assert orchestrator.run(source="unused.csv", metadata_standard="name") is None
    assert orchestrator.last_trace_path == str(failed_trace)
    failed_events = [json.loads(line) for line in failed_trace.read_text().splitlines()]
    assert failed_events[-2]["event"] == "planning_error"
    assert failed_events[-1]["event"] == "run_end"
