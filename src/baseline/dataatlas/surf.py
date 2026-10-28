from __future__ import annotations

from functools import lru_cache
from typing import Any

from data_dictionary_generator.description_generator import _post_process_response

from src.baseline.autoddg.llm.surf import create_surf_openai_client
from src.config import get_surf_extra_body


DEFAULT_SYSTEM_MESSAGE = (
    "You are a precise metadata expert. Write extremely concise, factual "
    "descriptions (max 15 words)."
)


@lru_cache(maxsize=1)
def _surf_client() -> Any:
    """Reuse one OpenAI-compatible SURF client inside each worker process."""
    return create_surf_openai_client()


def run_surf_llm(
    prompt: str,
    model: str,
    system_message: str | None = None,
    is_json: bool = False,
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
) -> str:
    """DataAtlas-compatible LLM function backed by SURF/Willma.

    DataAtlas routes non-OpenAI model names to Ollama. This adapter preserves
    its prompts and post-processing while replacing only that transport layer.
    """
    requested_tokens = max_output_tokens or (1000 if system_message else 50)
    # SURF's default-text-large alias can consume a small completion budget
    # internally and return finish_reason="length" with empty visible content.
    # Its output is still constrained to 15 words by DataAtlas post-processing.
    if model.lower() == "default-text-large":
        requested_tokens = max(requested_tokens, 256)

    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": requested_tokens,
    }
    extra_body = get_surf_extra_body(model)
    if extra_body:
        request["extra_body"] = extra_body

    response = _surf_client().chat.completions.create(**request)
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text and choice.finish_reason == "length":
        request["max_tokens"] = max(requested_tokens * 2, 512)
        response = _surf_client().chat.completions.create(**request)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
    if not text:
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        raise RuntimeError(
            "SURF returned an empty DataAtlas response "
            f"for {model} (finish_reason={choice.finish_reason!r}, "
            f"completion_tokens={completion_tokens!r})"
        )
    return text if is_json else _post_process_response(text)
