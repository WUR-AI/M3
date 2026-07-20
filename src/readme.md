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
