"""
Step-level execution with parallel players and debate.

This module implements the core logic for executing a single plan step:
1. Spawn multiple players (based on topology)
2. Each player executes the task in parallel
3. Players debate (critique and revise) their results
4. A synthesizer consolidates the final result

Uses the unified ExecutionContext abstraction for all data access.

The execution flow is:
    execute_parallel → critique → revise → [loop or synthesize]
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Type

from pydantic import BaseModel
from langgraph.graph import StateGraph, END

from src.core.state import StepExecutionState, PlayerResult, DebateEntry
from src.core.tool_trace import tool_result_is_error
from ..players import Player, create_player_from_config, PLAYER_CONFIGS
from ..tools.context_tools import filter_tools_by_context_type


def _parallel_map(players: List[Player], run):
    """Run blocking, independent model calls concurrently in player order."""
    if len(players) == 1:
        return [run(players[0])]
    with ThreadPoolExecutor(max_workers=len(players)) as pool:
        return list(pool.map(run, players))


# ===================================================================
#  NODE FUNCTIONS
# ===================================================================

def execute_parallel_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Execute the task with all players in parallel.
    
    Each player independently works on the same task using their tools
    and perspective. Results are collected for the debate phase.
    
    Uses ExecutionContext for unified data access.
    """
    logging.info(f"--- STEP {state['step_index']}: PARALLEL EXECUTION ---")
    logging.info(f"Task: {state['task']}")
    logging.info(f"Players: {len(state['players'])}")
    
    players: List[Player] = state["players"]
    task = state["task"]
    context_key = state["context_key"]
    context_info = state["context_info"]
    target_resources = state.get("target_resources", [])
    workspace = state["workspace"]
    input_mappings = state["input_mappings"]
    tool_tracer = state.get("tool_tracer")
    
    is_multi_csv = context_info.get("is_multi_csv", False)
    
    if is_multi_csv:
        logging.info(
            f"  Multi-CSV mode: {len(context_info.get('resources', []))} resources"
        )
        if target_resources:
            logging.info(f"  Target resources: {target_resources}")
    
    recorder = state.get("recorder")

    def run_player(player: Player):
        try:
            result = player.execute_task(
                task=task,
                context_key=context_key,
                context_info=context_info,
                workspace=workspace,
                inputs=input_mappings,
                target_resources=target_resources,
                tool_tracer=tool_tracer,
                recorder=recorder,
                step_index=state["step_index"],
            )
            player_result: PlayerResult = {
                "player_name": player.name,
                "task": task,
                "tool_results": result.get("tool_results", {}),
                "tool_trace": result.get("tool_trace", []),
                "analysis": result.get("analysis", ""),
                "success": result.get("success", True)
            }
            logging.info(f"  Player '{player.name}' completed execution")
            analysis_preview = result.get("analysis", "")[:200].replace('\n', ' ')
            if len(result.get("analysis", "")) > 200:
                analysis_preview += "..."
            logging.info(f"    Output: {analysis_preview}")
            return player_result, result.get("messages", [])
        except Exception as e:
            logging.error(f"  Player '{player.name}' failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            if recorder is not None:
                recorder.record(
                    "agent_error", step_index=state["step_index"],
                    agent=player.name, phase="initial", error=str(e),
                )
            return {
                "player_name": player.name,
                "task": task,
                "tool_results": {},
                "tool_trace": [],
                "analysis": f"Error: {str(e)}",
                "success": False
            }, []

    outcomes = _parallel_map(players, run_player)
    player_results = [result for result, _ in outcomes]
    agent_messages = {
        player.name: messages for player, (_, messages) in zip(players, outcomes)
    }
    initial_debate_entries: List[DebateEntry] = [
        {"round": 0, "player_name": result["player_name"],
         "entry_type": "initial_work", "content": result["analysis"]}
        for result in player_results if result["success"]
    ]
    
    return {
        "player_results": player_results,
        "agent_messages": agent_messages,
        "debate_log": state.get("debate_log", []) + initial_debate_entries,
        "current_debate_round": 0,
        "error": None if any(r["success"] for r in player_results) else "All players failed",
    }


def critique_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Each player critiques the work of other players.
    
    Players review each other's analyses and provide constructive feedback
    to improve the overall quality of the result.
    """
    current_round = state["current_debate_round"]
    logging.info(f"--- STEP {state['step_index']}: CRITIQUE (Round {current_round}) ---")
    
    players: List[Player] = state["players"]
    task = state["task"]
    debate_log = state["debate_log"]
    
    current_results = {
        result["player_name"]: result for result in state["player_results"]
        if result["success"]
    }
    active_players = [p for p in players if p.name in current_results]
    recorder = state.get("recorder")

    def critique_player(player: Player):
        other_work = {}
        for name, result in current_results.items():
            if name == player.name:
                continue
            evidence = {
                key: value for key, value in result["tool_results"].items()
                if not tool_result_is_error(value)
            }
            other_work[name] = (
                result["analysis"]
                + "\nVerified tool results: " + json.dumps(evidence, default=str)
            )
        if not other_work:
            return None
        try:
            critique = player.critique_work(
                task=task, other_players_work=other_work,
                recorder=recorder, step_index=state["step_index"],
                round_index=current_round,
                messages=state["agent_messages"][player.name],
            )
            entry: DebateEntry = {
                "round": current_round,
                "player_name": player.name,
                "entry_type": "critique",
                "content": critique["analysis"],
            }
            return entry, critique["messages"]
        except Exception as e:
            logging.error(f"  Player '{player.name}' critique failed: {str(e)}")
            if recorder is not None:
                recorder.record(
                    "agent_error", step_index=state["step_index"],
                    agent=player.name, phase=f"critique_{current_round + 1}",
                    error=str(e),
                )
            return None

    outcomes = _parallel_map(active_players, critique_player) if active_players else []
    new_entries: List[DebateEntry] = [outcome[0] for outcome in outcomes if outcome]
    agent_messages = dict(state["agent_messages"])
    for player, outcome in zip(active_players, outcomes):
        if outcome:
            agent_messages[player.name] = outcome[1]
    return {"debate_log": debate_log + new_entries, "agent_messages": agent_messages}


def revise_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Each player revises their work based on critiques.
    
    Players incorporate feedback from critiques to improve their analysis.
    """
    current_round = state["current_debate_round"]
    next_round = current_round + 1
    logging.info(f"--- STEP {state['step_index']}: REVISION (Round {next_round}) ---")
    
    players: List[Player] = state["players"]
    task = state["task"]
    debate_log = state["debate_log"]
    
    prior_results = {r["player_name"]: r for r in state["player_results"]}
    active_players = [p for p in players if prior_results[p.name]["success"]]
    recorder = state.get("recorder")

    def revise_player(player: Player):
        previous = prior_results[player.name]
        critiques = [
            entry["content"] for entry in debate_log
            if entry["round"] == current_round
            and entry["entry_type"] == "critique"
            and entry["player_name"] != player.name
        ]
        try:
            revised = player.revise_work(
                task=task,
                messages=state["agent_messages"][player.name],
                critiques=critiques,
                context_key=state["context_key"],
                tool_results=previous["tool_results"],
                tool_trace=previous["tool_trace"],
                tool_tracer=state.get("tool_tracer"),
                recorder=recorder,
                step_index=state["step_index"],
                round_index=next_round,
                required_resources=(
                    state["target_resources"] or state["context_info"].get("resources", [])
                ) if state["context_info"].get("is_multi_csv") else None,
            )
            result: PlayerResult = {
                "player_name": player.name,
                "task": task,
                "tool_results": revised["tool_results"],
                "tool_trace": revised["tool_trace"],
                "analysis": revised["analysis"],
                "success": True
            }
            logging.info(f"  Player '{player.name}' revised their work")
            return result, revised["messages"]
        except Exception as e:
            logging.error(f"  Player '{player.name}' revision failed: {str(e)}")
            if recorder is not None:
                recorder.record(
                    "agent_error", step_index=state["step_index"],
                    agent=player.name, phase=f"revision_{next_round}",
                    error=str(e),
                )
            return {**previous, "success": False, "analysis": f"Error: {e}"}, state["agent_messages"][player.name]

    outcomes = _parallel_map(active_players, revise_player) if active_players else []
    changed = {p.name: result for p, (result, _) in zip(active_players, outcomes)}
    updated_results = [changed.get(p.name, prior_results[p.name]) for p in players]
    agent_messages = dict(state["agent_messages"])
    for player, (_, messages) in zip(active_players, outcomes):
        agent_messages[player.name] = messages
    new_entries: List[DebateEntry] = [
        {"round": next_round, "player_name": r["player_name"],
         "entry_type": "revised_work", "content": r["analysis"]}
        for r in updated_results if r["success"]
    ]
    return {
        "debate_log": debate_log + new_entries,
        "current_debate_round": next_round,
        "player_results": updated_results,
        "agent_messages": agent_messages,
        "error": None if any(r["success"] for r in updated_results) else "All players failed during revision",
    }


def synthesize_node(state: StepExecutionState) -> Dict[str, Any]:
    """
    Synthesize all player results into a consolidated output.
    
    The synthesizer combines the best insights from all players
    and resolves any conflicts to produce a final result.
    
    If output_schema is provided in state, uses structured output
    to return a validated Pydantic model instead of a string.
    """
    logging.info(f"--- STEP {state['step_index']}: SYNTHESIS ---")
    
    task = state["task"]
    player_results = [r for r in state["player_results"] if r["success"]]
    expected_outputs = state["expected_outputs"]
    output_schema: Optional[Type[BaseModel]] = state.get("output_schema")
    
    if output_schema:
        logging.info(f"  Using structured output with schema: {output_schema.__name__}")
    
    if not player_results:
        return {
            "error": "No successful player result to synthesize",
            "consolidated_result": None,
            "produced_artifacts": {},
        }

    successful_names = {result["player_name"] for result in player_results}
    synthesizer = next(
        player for player in state["players"] if player.name in successful_names
    )

    try:
        # Prepare results for synthesis
        results_for_synthesis = [
            {
                "player": r["player_name"],
                "analysis": r["analysis"],
                "tool_results": {
                    key: value for key, value in r["tool_results"].items()
                    if not tool_result_is_error(value)
                },
            }
            for r in player_results
        ]
        
        # Synthesize with optional structured output
        recorder = state.get("recorder")
        if len(player_results) == 1 and output_schema is None:
            consolidated = player_results[0]["analysis"]
        else:
            consolidated = synthesizer.synthesize_results(
                task=task,
                all_results=results_for_synthesis,
                output_schema=output_schema,
                recorder=recorder,
                step_index=state["step_index"],
                conversation=state["agent_messages"][synthesizer.name],
            )
        if not consolidated or (isinstance(consolidated, str) and not consolidated.strip()):
            raise ValueError("Synthesis returned no result")
        if recorder is not None:
            recorder.record(
                "step_synthesis", step_index=state["step_index"],
                candidates=results_for_synthesis,
                output=consolidated.model_dump(mode="json") if isinstance(consolidated, BaseModel) else consolidated,
            )
        
        # Convert Pydantic model to dict for artifact storage if needed
        if output_schema and isinstance(consolidated, BaseModel):
            artifact_value = consolidated.model_dump(by_alias=True)
            logging.info("  Structured output validated successfully")
        else:
            artifact_value = consolidated
        
        # Create artifacts from the consolidated result
        produced_artifacts = {}
        for output_name in expected_outputs:
            produced_artifacts[output_name] = artifact_value
        
        logging.info(f"  Synthesis complete. Produced artifacts: {list(produced_artifacts.keys())}")
        # Log a preview of the synthesized output
        if isinstance(artifact_value, dict):
            preview = str(artifact_value)[:200].replace('\n', ' ')
        else:
            preview = str(artifact_value)[:200].replace('\n', ' ')
        if len(str(artifact_value)) > 200:
            preview += "..."
        logging.info(f"    Synthesized output: {preview}")
        
        return {
            "consolidated_result": consolidated,
            "produced_artifacts": produced_artifacts
        }
        
    except Exception as e:
        error_msg = f"Synthesis failed: {str(e)}"
        logging.error(f"  {error_msg}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "error": error_msg,
            "consolidated_result": None,
            "produced_artifacts": {}
        }


def debate_router(state: StepExecutionState) -> str:
    """
    Decide whether to continue debate or synthesize.
    
    Routes to:
    - 'critique_node': If more debate rounds needed
    - 'synthesize_node': If max rounds reached or single player
    - '__end__': If there's an error
    """
    if state.get("error"):
        logging.error(f"Error detected, ending step: {state['error']}")
        return "__end__"
    
    current_round = state["current_debate_round"]
    max_rounds = state["max_debate_rounds"]
    num_players = sum(1 for result in state["player_results"] if result["success"])
    
    # No debate is possible with fewer than two successful players.
    if num_players <= 1:
        logging.info("Single player, skipping debate")
        return "synthesize_node"
    
    # Check if we should continue debating
    if current_round < max_rounds:
        logging.info(f"Debate round {current_round}/{max_rounds}, continuing to critique")
        return "critique_node"
    else:
        logging.info(f"Max debate rounds ({max_rounds}) reached, synthesizing")
        return "synthesize_node"


# ===================================================================
#  GRAPH CONSTRUCTION
# ===================================================================

def get_step_execution_graph():
    """
    Constructs and compiles the StateGraph for step execution.
    
    The graph flow is:
        execute_parallel → [debate_router] → critique → revise → [debate_router] → ... → synthesize
    
    Returns:
        A compiled langgraph application for executing a single step.
    """
    graph = StateGraph(StepExecutionState)
    
    # Add nodes
    graph.add_node("execute_parallel_node", execute_parallel_node)
    graph.add_node("critique_node", critique_node)
    graph.add_node("revise_node", revise_node)
    graph.add_node("synthesize_node", synthesize_node)
    
    # Set entry point
    graph.set_entry_point("execute_parallel_node")
    
    # Define edges
    # After parallel execution, route based on debate config
    graph.add_conditional_edges(
        "execute_parallel_node",
        debate_router,
        {
            "critique_node": "critique_node",
            "synthesize_node": "synthesize_node",
            "__end__": END
        }
    )
    
    # After critique, always revise
    graph.add_edge("critique_node", "revise_node")
    
    # After revision, route again
    graph.add_conditional_edges(
        "revise_node",
        debate_router,
        {
            "critique_node": "critique_node",
            "synthesize_node": "synthesize_node",
            "__end__": END
        }
    )
    
    # After synthesis, end
    graph.add_edge("synthesize_node", END)
    
    return graph.compile()


# ===================================================================
#  HELPER FUNCTIONS
# ===================================================================

def create_step_state(
    step_index: int,
    step_dict: Dict[str, Any],
    context: Any,
    context_key: str,
    workspace: Dict[str, Any],
    metadata_standard: str,
    players_per_step: int,
    debate_rounds: int,
    player_pool: List[str],
    output_schema: Optional[Type[BaseModel]] = None,
    tool_tracer: Optional[Any] = None,
    recorder: Optional[Any] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> StepExecutionState:
    """
    Create the initial state for executing a step.
    
    Args:
        step_index: Index of this step in the plan
        step_dict: The step dictionary from the plan
        context: The ExecutionContext to analyze
        context_key: Key for the ExecutionContext in the tool registry
        workspace: Current workspace with artifacts
        metadata_standard: The metadata standard to follow
        players_per_step: Number of players to spawn
        debate_rounds: Number of debate rounds
        player_pool: List of player role names available
        output_schema: Optional Pydantic model for structured output (typically for final step)
        
    Returns:
        Initialized StepExecutionState
    """
    # Get the player role specified in the plan for this step
    specified_player_role = step_dict.get("player", "")
    
    # Create player instances - all of the same type specified in the plan
    players = []
    
    # Determine which player role to use
    if specified_player_role and specified_player_role in PLAYER_CONFIGS:
        role_to_use = specified_player_role
    elif player_pool:
        role_to_use = player_pool[0]  # Fallback to first in pool
    else:
        role_to_use = "data_analyst"  # Ultimate fallback
    
    # Create multiple instances of the same player type
    config = PLAYER_CONFIGS.get(role_to_use, {}).copy()
    if model_name is not None:
        config["model_name"] = model_name
    config["tools"] = filter_tools_by_context_type(
        config.get("tools", []), context.context_type
    )
    for i in range(players_per_step):
        player = create_player_from_config(
            config, name=f"{role_to_use}_{i+1}", role_key=role_to_use,
            provider=provider,
        )
        players.append(player)
    
    logging.info(f"  Created {players_per_step} '{role_to_use}' player(s)")
    
    # Use the first player as synthesizer
    synthesizer = players[0] if players else None
    
    target_resources = step_dict.get("target_resources", [])
    
    return StepExecutionState(
        step_index=step_index,
        task=step_dict.get("task", ""),
        player_name=step_dict.get("player", ""),
        rationale=step_dict.get("rationale", ""),
        input_mappings=step_dict.get("inputs", {}),
        expected_outputs=step_dict.get("outputs", []),
        target_resources=target_resources,
        context_key=context_key,
        context_info=context.to_dict(),
        workspace=workspace,
        metadata_standard=metadata_standard,
        tool_tracer=tool_tracer,
        recorder=recorder,
        players=players,
        synthesizer=synthesizer,
        max_debate_rounds=debate_rounds,
        current_debate_round=0,
        player_results=[],
        agent_messages={},
        debate_log=[],
        output_schema=output_schema,
        consolidated_result=None,
        produced_artifacts={},
        error=None,
    )
