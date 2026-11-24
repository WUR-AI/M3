"""
Defines the state schemas for the multi-agent system.

This module contains TypedDict definitions for:

1. StepExecutionState: State for executing a single plan step with parallel players
2. PlanExecutionState: Top-level state for executing an entire plan

Uses the unified ExecutionContext abstraction for all data access.
"""

from typing import TypedDict, List, Dict, Any, Optional, Type

from pydantic import BaseModel

from src.core.schemas import Task


class PlayerResult(TypedDict):
    """Result from a single player's execution."""
    player_name: str
    task: str
    tool_results: Dict[str, Any]
    tool_trace: List[Dict[str, Any]]
    analysis: str
    success: bool


class DebateEntry(TypedDict):
    """
    A single entry in the debate log.

    Attributes:
        entry_type: 'initial_work', 'critique', 'revised_work'.
    """
    round: int
    player_name: str
    entry_type: str
    content: str


class StepExecutionState(TypedDict):
    """
    State for executing a single plan step with multiple parallel players.
    
    This state is used by the step-level debate graph where:

    1. Multiple players execute the same task in parallel
    2. Players debate (critique and revise) their results
    3. One of the players synthesizes the final result using their role expertise
    
    Uses the unified :class:`~src.context.ExecutionContext` abstraction for all data access. 
    
    Attributes:
        step_index: Index of this step in the plan.
        task: The task description.
        player_name: The player type for this step (from plan).
        rationale: Why this step is needed.
        input_mappings: Maps param names to artifact names.
        expected_outputs: Artifact names this step should produce.
        target_resources: Which resources this step targets (empty = all).
        context_key: Key to registered ExecutionContext in tool registry.
        context_info: Serialized ExecutionContext info.
        workspace: Artifacts from previous steps.
        metadata_standard: The metadata standard to follow.
        players: List of Player instances for this step.
        synthesizer: Player instance for synthesis (one of the players).
        max_debate_rounds: Maximum debate rounds for this step.
        current_debate_round: Current debate round (starts at 1).
        tool_tracer: Shared tool trace for this plan execution.
        recorder: Durable run trace.
        player_results: Results from parallel execution.
        agent_messages: Each agent's uninterrupted conversation.
        debate_log: Log of debate entries.
        output_schema: Pydantic schema for structured output (final step only).
        consolidated_result: Final synthesized result (str or BaseModel).
        produced_artifacts: Artifacts produced by this step.
        error: Error message if something went wrong.
    """
    # --- Step Configuration ---
    step_index: int
    task: str
    player_name: str
    rationale: str
    input_mappings: Dict[str, str]
    expected_outputs: List[str]
    target_resources: List[str]
    
    # --- Execution Context ---
    context_key: str
    context_info: Dict[str, Any]
    workspace: Dict[str, Any]
    metadata_standard: str
    tool_tracer: Optional[Any]
    recorder: Optional[Any]
    
    # --- Player Configuration ---
    players: List[Any]
    synthesizer: Any
    
    # --- Debate Configuration ---
    max_debate_rounds: int
    current_debate_round: int
    
    # --- Dynamic State ---
    player_results: List[PlayerResult]
    agent_messages: Dict[str, List[Any]]
    debate_log: List[DebateEntry]
    
    # --- Structured Output ---
    output_schema: Optional[Type[BaseModel]]
    
    # --- Output ---
    consolidated_result: Optional[Any]
    produced_artifacts: Dict[str, Any]
    error: Optional[str]


class PlanExecutionState(TypedDict):
    """
    Top-level state for executing an entire plan.
    
    This state tracks progress through all steps in a plan,
    accumulating artifacts in the workspace as steps complete.
    
    Uses the unified :class:`~src.context.ExecutionContext` abstraction for all data access.

    Attributes:
        plan_steps: The steps from the Plan object.
        current_step_index: Which step we're on (0-indexed).
        context_key: Key to registered ExecutionContext in tool registry.
        context_info: Serialized ExecutionContext info.
        metadata_standard: The metadata standard to follow.
        topology_name: Name of the execution topology.
        players_per_step: How many players per step.
        debate_rounds: Debate rounds per step.
        player_pool: Available player role names.
        workspace: Artifacts accumulated from all steps.
        step_results: Results from each completed step.
        resource_metadata: Per-resource metadata results.
        discovered_relationships: Discovered relationships.
        final_output: Final metadata output.
        error: Error if execution failed.
    """
    # --- Plan Configuration ---
    plan_steps: List[Task]
    current_step_index: int
    
    # --- Execution Context ---
    context_key: str
    context_info: Dict[str, Any]
    metadata_standard: str
    
    # --- Topology Configuration ---
    topology_name: str
    players_per_step: int
    debate_rounds: int
    player_pool: List[str]
    
    # --- Accumulated State ---
    workspace: Dict[str, Any]
    step_results: List[Dict[str, Any]]
    
    # --- Context Results ---
    resource_metadata: Dict[str, Any]
    discovered_relationships: List[Dict[str, Any]]
    
    # --- Output ---
    final_output: Optional[Dict[str, Any]]
    error: Optional[str]
