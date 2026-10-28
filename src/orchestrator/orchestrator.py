"""
Main Orchestrator - Coordinates planning and execution.
"""

import logging
import uuid
from typing import Dict, List, Optional, Union

from langchain_core.output_parsers import PydanticOutputParser

from src.core.schemas import Plan, Task, ExecutionResult
from src.core.run_trace import RunTraceRecorder, message_record

from ..config import (
    DEFAULT_TOPOLOGY,
    LLM_PROVIDER,
    PLANNING_TEMPERATURE,
    create_llm,
)
from ..context import ContextType, ExecutionContext, create_context
from ..context.context_classifier import classify_context_type
from ..players import (
    METADATA_OUTPUT_PLAYER_NAMES,
    PLAYER_CONFIGS,
    create_player_from_config,
)
from ..tools.context_tools import (
    register_context,
    unregister_context,
    filter_tools_by_context_type,
)
from ..topology import EXECUTION_TOPOLOGIES
from .plan_executor import PlanExecutor
from .prompts import get_multi_csv_planning_prompt, get_single_csv_planning_prompt
from ..standards.standards import planning_metadata_standard
from .utils import (
    validate_plan_completion, validate_plan_dataflow,
    validate_plan_tool_compatibility,
)


class Orchestrator:
    """
    The main orchestrator that coordinates plan generation and execution.
    """
    MULTI_CONTEXT_TYPES = {
        ContextType.MULTI_CSV,
    }

    def __init__(
        self,
        topology_name: str = DEFAULT_TOPOLOGY,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        topology_name = topology_name or DEFAULT_TOPOLOGY
        temperature = temperature if temperature is not None else PLANNING_TEMPERATURE
        provider = provider or LLM_PROVIDER

        if topology_name not in EXECUTION_TOPOLOGIES:
            available = list(EXECUTION_TOPOLOGIES.keys())
            raise ValueError(
                f"Unknown topology '{topology_name}'. Available: {available}"
            )

        self.topology_name = topology_name
        self.topology = EXECUTION_TOPOLOGIES[topology_name]
        self.provider = provider
        self.model_name = model_name
        self.last_trace_path: Optional[str] = None

        self.llm = create_llm(
            model_name=model_name, temperature=temperature, provider=provider
        )
        self.parser = PydanticOutputParser(pydantic_object=Plan)

        self.executor = PlanExecutor(
            topology_name=topology_name, model_name=model_name, provider=provider,
        )

        logging.info(f"Orchestrator initialized with topology: {topology_name}")

    def _classify_context_for_planning(self, context: ExecutionContext) -> ContextType:
        """
        Classify context for prompt routing using context classifier when possible.
        Falls back to context.context_type.
        """
        if hasattr(context, "get_all_file_paths"):
            try:
                file_paths = list(context.get_all_file_paths().values())
                return classify_context_type(file_paths)
            except Exception:
                pass
        return context.context_type

    def _get_effective_player_pool(self, context: ExecutionContext = None) -> list:
        player_pool = list(self.topology.get("player_pool", []))

        # Always add metadata output players for final step generation
        for player_name in METADATA_OUTPUT_PLAYER_NAMES:
            if player_name not in player_pool:
                player_pool.append(player_name)
                logging.info(
                    "Auto-added '%s' for final metadata generation", player_name
                )

        if context:
            classified_type = self._classify_context_for_planning(context)
            is_multi_context = classified_type in self.MULTI_CONTEXT_TYPES
        else:
            is_multi_context = False

        if is_multi_context:
            if "relationship_analyst" not in player_pool:
                player_pool.append("relationship_analyst")
                logging.info(
                    "Auto-added 'relationship_analyst' for multi-context type"
                )

        return player_pool

    def _validate_plan(
        self, plan: Plan, context: ExecutionContext,
        metadata_standard_name: Optional[str] = None,
        recorder: Optional[RunTraceRecorder] = None,
    ) -> bool:
        """
        Validate generated plan before execution.

        Keeps validation separate from plan generation so generate_plan stays pure.
        """
        plan_dict = plan.to_dict_list()

        dataflow_ok, dataflow_msg = validate_plan_dataflow(plan_dict)
        if not dataflow_ok:
            logging.error("Plan dataflow validation failed: %s", dataflow_msg)
            if recorder is not None:
                recorder.record("plan_validation", success=False, reason=dataflow_msg)
            return False

        completion_ok, completion_msg = validate_plan_completion(
            plan_dict, metadata_standard_name, context.resources
        )
        if not completion_ok:
            logging.error("Plan completion validation failed: %s", completion_msg)
            if recorder is not None:
                recorder.record("plan_validation", success=False, reason=completion_msg)
            return False

        allowed_players = set(self._get_effective_player_pool(context))
        tools_ok, tools_msg = validate_plan_tool_compatibility(
            plan=plan_dict,
            context_type=context.context_type,
            allowed_players=allowed_players,
        )
        if not tools_ok:
            logging.error("Plan tool validation failed: %s", tools_msg)
            if recorder is not None:
                recorder.record("plan_validation", success=False, reason=tools_msg)
            return False

        if recorder is not None:
            recorder.record("plan_validation", success=True)
        return True

    def _generate_player_manifest(self, context: ExecutionContext = None) -> str:
        player_pool = self._get_effective_player_pool(context)

        manifest_parts = []
        for role_name in player_pool:
            if role_name in PLAYER_CONFIGS:
                config = PLAYER_CONFIGS[role_name].copy()
                if self.model_name is not None:
                    config["model_name"] = self.model_name
                if context is not None:
                    config["tools"] = filter_tools_by_context_type(
                        config.get("tools", []), context.context_type
                    )
                player = create_player_from_config(
                    config, name=role_name, role_key=role_name,
                    provider=self.provider,
                )
                manifest_parts.append(player.get_tool_manifest())

        return "\n\n".join(manifest_parts)

    def _generate_context_info(self, context: ExecutionContext) -> str:
        info_parts = [
            f"Context Name: {context.name}",
            f"Context Type: {context.context_type.value}",
            f"Multi-CSV: {context.is_multi_csv}",
            f"Resources: {', '.join(context.resources)}",
        ]

        if context.description:
            info_parts.insert(1, f"Description: {context.description}")

        info_parts.append("\nResource Details:")
        for resource in context.resources:
            try:
                resource_info = context.get_resource_info(resource)
                info_parts.append(
                    f"  - {resource}: {resource_info.item_count} items, "
                    f"{resource_info.field_count} fields ({', '.join(resource_info.field_names[:5])}{'...' if len(resource_info.field_names) > 5 else ''})"
                )
            except Exception:
                info_parts.append(f"  - {resource}: (info unavailable)")

        relationships = context.get_relationships()
        if relationships:
            info_parts.append("\nDiscovered Relationships:")
            for rel in relationships[:5]:
                info_parts.append(
                    f"  - {rel.from_resource}.{rel.from_field} -> "
                    f"{rel.to_resource}.{rel.to_field} ({rel.relationship_type})"
                )
            if len(relationships) > 5:
                info_parts.append(f"  ... and {len(relationships) - 5} more")

        return "\n".join(info_parts)

    def generate_plan(
        self,
        context: ExecutionContext,
        metadata_standard: str,
        metadata_standard_name: Optional[str] = None,
        recorder: Optional[RunTraceRecorder] = None,
    ) -> Optional[Plan]:
        classified_type = self._classify_context_for_planning(context)
        is_multi_context = classified_type in self.MULTI_CONTEXT_TYPES

        # A fixed benchmark standard has a known dependency graph. Keeping it
        # explicit avoids ambiguous Croissant routing in the generic planner.
        if metadata_standard_name == "pangaea_descriptive_technical_standard" and not is_multi_context:
            plan = Plan(steps=[
                Task(task="Inspect all columns and representative values with get_columns_with_first_row",
                     player="dataset_schema_preview", rationale="Capture the source schema and values",
                     outputs=["schema_preview"]),
                Task(task="Profile fields with get_field_statistics",
                     player="data_analyst", rationale="Ground descriptive fields in distributions",
                     inputs={"schema": "schema_preview"}, outputs=["field_statistics"]),
                Task(task="Detect spatial and temporal columns and calculate full extents",
                     player="spatial_temporal_specialist", rationale="Ground coverage and resolution",
                     inputs={"schema": "schema_preview"}, outputs=["spatial_temporal_data"]),
                Task(task="Produce the seven requested descriptive and technical fields",
                     player="metadata_generator", rationale="Combine verified source evidence",
                     inputs={"metadata_standard": "metadata_standard", "schema": "schema_preview",
                             "statistics": "field_statistics", "spatial_temporal": "spatial_temporal_data"},
                     outputs=["metadata_output"]),
            ])
            if recorder is not None:
                recorder.record("plan", plan=plan.model_dump(mode="json"), source="fixed_standard")
            return plan

        logging.info("[ui] planning")
        logging.info("=" * 60)
        logging.info("GENERATING PLAN")
        logging.info(f"Context: {context.name}")
        logging.info(f"Context type: {context.context_type.value}")
        logging.info(f"Classified planning type: {classified_type.value}")
        logging.info(f"Resources: {context.resources}")
        logging.info(f"Is multi-context: {is_multi_context}")
        if metadata_standard_name:
            logging.info(f"Metadata standard name: {metadata_standard_name}")
        logging.info("=" * 60)

        manifest = self._generate_player_manifest(context)
        context_info = self._generate_context_info(context)
        planner_standard = planning_metadata_standard(
            metadata_standard, metadata_standard_name
        )

        logging.info("Prepared planning inputs.")
        logging.info("  Context summary length: %d chars", len(context_info))
        logging.info("  Player manifest length: %d chars", len(manifest))
        logging.info("  Resource count: %d", len(context.resources))

        try:
            format_instructions = self.parser.get_format_instructions()
            planning_prompt = (
                get_multi_csv_planning_prompt() if is_multi_context
                else get_single_csv_planning_prompt()
            )

            if is_multi_context:
                prompt_inputs = {
                    "dataset_info": context_info,
                    "dataset_name": context.name,
                    "table_names": ", ".join(context.resources),
                    "file_type": context.context_type.value.upper(),
                    "available_players": manifest,
                    "metadata_standard": planner_standard,
                    "format_instructions": format_instructions,
                }

            else:
                prompt_inputs = {
                    "file_type": context.context_type.value.upper(),
                    "available_players": manifest,
                    "metadata_standard": planner_standard,
                    "format_instructions": format_instructions,
                }

            messages = planning_prompt.format_messages(**prompt_inputs)
            if recorder is not None:
                recorder.record(
                    "model_input", phase="planning", agent="orchestrator",
                    messages=[message_record(msg) for msg in messages],
                )
            raw_output = self.llm.invoke(messages)
            if recorder is not None:
                recorder.record(
                    "model_output", phase="planning", agent="orchestrator",
                    message=message_record(raw_output),
                )
            generated_plan = self.parser.invoke(raw_output)
            if recorder is not None:
                recorder.record("plan", plan=generated_plan.model_dump(mode="json"))

            logging.info("Plan generated successfully!")
            logging.info(f"Number of steps: {len(generated_plan.steps)}")
            for i, step in enumerate(generated_plan.steps):
                target_info = (
                    f" (resources: {step.target_resources})" if step.target_resources else ""
                )
                logging.info(
                    f"  Step {i + 1}: {step.task} (player: {step.player}){target_info}"
                )

            return generated_plan

        except Exception as e:
            logging.error(f"Plan generation failed: {e}")
            if recorder is not None:
                recorder.record("planning_error", error=str(e))
            return None

    def execute_plan(
        self, 
        plan: Plan, 
        context: ExecutionContext, 
        metadata_standard: str,
        metadata_standard_name: Optional[str] = None,
        recorder: Optional[RunTraceRecorder] = None,
    ) -> ExecutionResult:
        context_key = f"ctx_{uuid.uuid4().hex[:8]}"
        register_context(context_key, context)

        effective_player_pool = self._get_effective_player_pool(context)

        try:
            return self.executor.execute(
                plan=plan,
                context=context,
                context_key=context_key,
                metadata_standard=metadata_standard,
                metadata_standard_name=metadata_standard_name,
                player_pool=effective_player_pool,
                recorder=recorder,
            )
        finally:
            unregister_context(context_key)
    
    # decorators for logging?
    def run(
        self,
        source: Union[str, List[str], Dict[str, str], ExecutionContext],
        metadata_standard: str,
        metadata_standard_name: Optional[str] = None,
        name: str = "context",
        **kwargs,
    ) -> Optional[ExecutionResult]:
        """
        Run the complete orchestration: generate plan and execute.
        
        Args:
            source: Data source (path, list of paths, dict, or ExecutionContext)
            metadata_standard: The metadata standard content (string template)
            metadata_standard_name: Optional name of the standard (e.g., "dublin_core", 
                                   "relational") for structured output. If provided,
                                   the final step will use Pydantic validation.
            name: Name for the context
            **kwargs: Additional arguments passed to create_context
            
        Returns:
            ExecutionResult with final metadata as structured dict
        """
        if isinstance(source, ExecutionContext):
            context = source
        else:
            context = create_context(source, name=name, **kwargs)

        recorder = RunTraceRecorder.create(context.name)
        self.last_trace_path = str(recorder.path)
        logging.info("Run trace: %s", recorder.path)
        recorder.record(
            "run_start", context=context.to_dict(), topology=self.topology_name,
            standard_name=metadata_standard_name, model_name=self.model_name,
            provider=self.provider,
        )

        logging.info("=" * 60)
        logging.info("STARTING ORCHESTRATION")
        logging.info(f"Context: {context.name}")
        logging.info(f"Type: {context.context_type.value}")
        logging.info(f"Resources: {context.resources}")
        if metadata_standard_name:
            logging.info(f"Metadata standard: {metadata_standard_name} (structured output)")
        logging.info("=" * 60)

        plan = self.generate_plan(
            context=context,
            metadata_standard=metadata_standard,
            metadata_standard_name=metadata_standard_name,
            recorder=recorder,
        )

        if plan is None:
            logging.error("Failed to generate plan. Aborting execution.")
            recorder.record("run_end", success=False, reason="planning_failed")
            return None

        if not self._validate_plan(plan, context, metadata_standard_name, recorder):
            logging.error("Generated plan failed validation. Aborting execution.")
            recorder.record("run_end", success=False, reason="invalid_plan")
            return None

        try:
            result = self.execute_plan(
                plan=plan,
                context=context,
                metadata_standard=metadata_standard,
                metadata_standard_name=metadata_standard_name,
                recorder=recorder,
            )
        except Exception as exc:
            recorder.record("run_end", success=False, error=str(exc))
            raise

        recorder.record("run_end", success=result.success, error=result.error)

        return result


def run_metadata_extraction(
    source: Union[str, List[str], Dict[str, str], ExecutionContext],
    metadata_standard: str,
    metadata_standard_name: Optional[str] = None,
    name: str = "context",
    topology_name: str = "default",
    **kwargs,
) -> Optional[ExecutionResult]:
    """
    Convenience function to run metadata extraction.
    
    Args:
        source: Data source (path, list of paths, dict, or ExecutionContext)
        metadata_standard: The metadata standard content (string template)
        metadata_standard_name: Optional name of the standard for structured output
        name: Name for the context
        topology_name: Name of the execution topology
        **kwargs: Additional arguments passed to create_context
        
    Returns:
        ExecutionResult with final metadata
    """
    orchestrator = Orchestrator(topology_name=topology_name)
    return orchestrator.run(
        source=source, 
        metadata_standard=metadata_standard, 
        metadata_standard_name=metadata_standard_name,
        name=name, 
        **kwargs
    )
