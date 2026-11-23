# TUI Usage

This TUI can be installed as a CLI and run from any folder.

## Install

From the repository root, install it in editable mode:

```bash
cd /home/com3dian/Github/metadata_agent
python -m pip install -e .
```

Editable install means changes you make in this repository take effect the next time you run the tool, without reinstalling in most cases.

## Run

After installation, launch the TUI from any folder with:

```bash
metadata-agent --tui
```

## When to Reinstall

Usually, code changes do not require reinstalling when using `-e`.

Run this again only when packaging or dependency configuration changes:

```bash
python -m pip install -e .
```
