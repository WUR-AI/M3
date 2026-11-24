# M³ core library

This folder contains the metadata extraction pipeline: orchestration, agents, data access, tools, and metadata standards.

## Installation

**Requirements:** Python 3.12+ and [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
make uv-setup
```

This creates a `.venv` virtual environment and installs core dependencies from [`pyproject.toml`](../pyproject.toml).

**Without Make:**

```bash
uv python pin 3.12
uv sync
```

**CLI (editable install):**

```bash
uv pip install -e .
```

After this, `metadata-agent` is available on your path.

**Optional dependency groups:**

```bash
uv sync --group docs    # Sphinx documentation
```

## LLM configuration

Create a `.env` file in the project root. Example for Google Gemini (default provider):

```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=your_api_key_here
```

Other providers (OpenAI, SURF/Willma) and model options are documented in [`docs/tutorial.md`](../docs/tutorial.md#llm-provider-configuration).

## Usage

```bash
# CLI
metadata-agent --source ./data/my_data.csv

# Terminal UI
metadata-agent --tui
```

For a full walkthrough, see [`docs/tutorial.md`](../docs/tutorial.md). For architecture details, see [`docs/architecture.md`](../docs/architecture.md).

## Inspecting a run

Every orchestration writes an append-only JSONL trace under `outputs/traces/`.
The returned `ExecutionResult.trace_path` gives the exact file. Each record
identifies the phase and agent and keeps the complete model input, model output,
tool arguments and result, critique, revision, synthesis, or error. A trace is
also kept when planning or a step fails. The file is created with owner-only
permissions because it can contain sampled source data.
If planning fails before an `ExecutionResult` exists, use
`Orchestrator.last_trace_path` to locate the trace.

Within a step, agents work concurrently and each keeps its own tool conversation
through the configured debate rounds. The final generator receives verified
tool observations from preceding steps. For iNaturalist coverage, full-data
extent results take precedence over conflicting generated values; corrections
are recorded in the trace.
