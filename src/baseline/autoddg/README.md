# AutoDDG baseline source

Copied from https://github.com/VIDA-NYU/AutoDDG at commit `3c59e1c466a5c0ad3e9126f8a09023815b437aef`.
Upstream code remains under Apache-2.0; see LICENSE.

## Local changes

- Relocated the package to `src.baseline.autoddg`.
- Updated absolute imports, example imports, and the prompt resource package path.
- Included prompts.yaml and LICENSE in the project's package data.
- Added a SURF/Willma client factory (`create_surf_openai_client`, `AutoDDG.from_surf`)
  so generation uses the same OpenAI-compatible endpoint as M³.
- `OpenAICompatibleClient` accepts default chat-completions kwargs (Willma
  `extra_body` / `enable_thinking`).
- Generation prompts and baseline behavior are otherwise unchanged.

## Running on SURF

Requires `SURF_API_KEY` and `SURF_API_BASE` in the project `.env` (same as M³).
The PANGAEA runner is `notebooks/related_work/autoddg_pangaea.ipynb`.
Each CSV is given 10 minutes; timeouts are recorded as
`{id}_autoddg.skipped.json` plus `run_log.jsonl` and skipped on resume.

```python
from src.baseline.autoddg import AutoDDG

pipe = AutoDDG.from_surf()  # uses LLM_MODEL; pass model_name=... to override
prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
```

Extra packages used by this baseline (already present in the `sam2` env):
`beartype`, `datamart-profiler`, `pyyaml`, `openai`.
