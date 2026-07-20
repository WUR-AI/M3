<div align="center">
<img width="6696" height="2456" alt="Frame 1(4)" src="https://github.com/user-attachments/assets/e8fe8071-8531-494b-9d65-942b25202486" />


# $M^3$
[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-lightgrey.svg)](https://github.com/com3dian/metadata_agent)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/com3dian/metadata_agent) <br>
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/com3dian/metadata_agent/pulls)
[![Issues](https://img.shields.io/github/issues/com3dian/metadata_agent)](https://github.com/com3dian/metadata_agent/issues)
![GitHub Tag](https://img.shields.io/github/v/tag/com3dian/metadata_agent)
[![test-main](https://github.com/WUR-AI/M3/actions/workflows/ci-uv.yml/badge.svg?branch=main)](https://github.com/WUR-AI/M3/actions/workflows/ci-uv.yml)
[![contributors](https://img.shields.io/github/contributors/com3dian/metadata_agent.svg)](https://github.com/com3dian/metadata_agent/graphs/contributors)

</div>
<br>

## 📝 Description

$M^3$ is a multi-agent LLM pipeline that plans and analyzes over tabular datasets to produce structured metadata. It is built based on the [MAST4Science template](https://github.com/com3dian/multi_agent_system_template).

<br>

## Where to find what

| Path | What you'll find |
|------|------------------|
| [`src/`](./src) | Core library: orchestration, agents, data access, and metadata standards |
| [`src/orchestrator/`](./src/orchestrator) | Planning and step execution (parallel agents, debate, synthesis) |
| [`src/players/`](./src/players) | Agent roles, prompts, and tool assignments |
| [`src/context/`](./src/context) | Unified data layer for CSV, SQLite, and multi-table inputs |
| [`src/tools/`](./src/tools) | Tools agents use to inspect datasets (schema, samples, relationships) |
| [`src/standards.py`](./src/standards.py) | Predefined metadata output formats (e.g. Dublin Core, relational) |
| [`src/topology.py`](./src/topology.py) | Execution topologies (player count, debate rounds) |
| [`src/main.py`](./src/main.py) | CLI entry point (`metadata-agent`) |
| [`src/tui/`](./src/tui) | Terminal UI for interactive metadata extraction |
| [`demo/`](./demo) | Streamlit app pages and workflow logic |
| [`demo_app.py`](./demo_app.py) | Streamlit demo entry point (`make demo`) |
| [`docs/`](./docs) | Tutorial, architecture overview, and deployment guides |
| [`examples/`](./examples) | Small scripts to test LLM connectivity and run the pipeline |
| [`tests/`](./tests) | Unit tests |
| [`data/`](./data) | Sample datasets for development and experiments |
| [`notebooks/`](./notebooks) | Exploratory notebooks for development |
| [`pyproject.toml`](./pyproject.toml) | Dependencies and package configuration |
| [`makefile`](./makefile) | Common commands (setup, docs, demo, test, lint) |

**Getting started**

- **Setup:** `make uv-setup` — create the virtual environment and install dependencies
- **Tutorial:** [`docs/tutorial.md`](./docs/tutorial.md) — full walkthrough from configuration to output
- **Architecture:** [`docs/architecture.md`](./docs/architecture.md) — how planning, execution, and debate fit together
- **CLI:** `metadata-agent --source ./data/my_data.csv`
- **Terminal UI:** `metadata-agent --tui`
- **Web demo:** `make demo`
- **Example script:** `python -m examples.generation`
