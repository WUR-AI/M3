from src.orchestrator import Orchestrator
from src.standards import METADATA_STANDARDS
from src.context.context_factory import create_context
from src.orchestrator.plan_executor import PlanExecutor
from src.tools.context_tools import register_context
from src.config import LLM_PROVIDER, PLANNING_TEMPERATURE, get_model_name
import json
import logging
from pathlib import Path
import os
from time import perf_counter
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.propagate = False

load_dotenv()

process_start = perf_counter()
step_start = process_start


def log_step_section(step_number: int, step_name: str) -> None:
    logger.info("")
    logger.info("------------------- Step %d: %s -------------------", step_number, step_name)


def log_step_timing(step_name: str, start: float) -> float:
    now = perf_counter()
    logger.info("[timer] %s: %.3fs", step_name, now - start)
    return now


# 1. Define source
log_step_section(1, "Define source")
source = {"data": os.getenv("DATA_FILE")}
topology_name = os.getenv("TOPOLOGY_NAME", "default")
step_start = log_step_timing("Define source", step_start)

# 2. Create context
log_step_section(2, "Create context")
context = create_context(source=source, name="my_dataset")
step_start = log_step_timing("Create context", step_start)

logger.info("------------------- Starting Plan Generation and Execution -------------------")
logger.info("Data source: %s", source["data"])
logger.info("Topology name: %s", topology_name)
logger.info("Model name: %s", get_model_name())
logger.info("LLM Provider: %s", LLM_PROVIDER)
logger.info("Planning temperature: %s", PLANNING_TEMPERATURE)
# 3. Generate plan
log_step_section(3, "Generate plan")
orchestrator = Orchestrator(
    topology_name=topology_name,
    model_name=get_model_name(),
    temperature=PLANNING_TEMPERATURE,
    provider=LLM_PROVIDER,
)
plan = orchestrator.generate_plan(
    context=context,
    metadata_standard=METADATA_STANDARDS["spatial_ecological"]
)
step_start = log_step_timing("Generate plan", step_start)
plan_payload = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else plan
logger.info("Generated Plan:\n%s", json.dumps(plan_payload, indent=2, default=str))


# 4. Execute
log_step_section(4, "Execute plan")
context_key = "ctx_my_dataset"
register_context(context_key, context)

executor = PlanExecutor(topology_name=topology_name)
result = executor.execute(
    plan=plan,
    context=context,
    context_key=context_key,
    metadata_standard=METADATA_STANDARDS["spatial_ecological"],
    metadata_standard_name="spatial_ecological"
)
step_start = log_step_timing("Execute plan", step_start)

# 5. Get metadata
log_step_section(5, "Write metadata")
metadata_output = result.final_workspace['metadata_output']
output_dir = Path(os.getenv("OUTPUT_DIR") or "output")
output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / f"metadata_{context.name}.json").open("w", encoding="utf-8") as f:
    json.dump(metadata_output, f, ensure_ascii=False, indent=2, default=str)

logger.info("Extracted Metadata:")
logger.info("%s", json.dumps(metadata_output, ensure_ascii=False, indent=2, default=str))
step_start = log_step_timing("Write metadata", step_start)
logger.info("[timer] Whole process: %.3fs", step_start - process_start)
logger.info("------------------- End of Execution -------------------")
