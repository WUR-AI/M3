from __future__ import annotations

from typing import Any

from beartype import beartype

from .base import LLMClient


def _merge_request_kwargs(
    default_kwargs: dict[str, Any], call_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Merge per-client defaults with a single call, including extra_body."""
    merged = {**default_kwargs, **call_kwargs}
    default_extra = dict(default_kwargs.get("extra_body") or {})
    call_extra = dict(call_kwargs.get("extra_body") or {})
    if default_extra or call_extra:
        extra_body = {**default_extra, **call_extra}
        default_ctk = dict(default_extra.get("chat_template_kwargs") or {})
        call_ctk = dict(call_extra.get("chat_template_kwargs") or {})
        if default_ctk or call_ctk:
            extra_body["chat_template_kwargs"] = {**default_ctk, **call_ctk}
        if extra_body:
            merged["extra_body"] = extra_body
        else:
            merged.pop("extra_body", None)
    return merged


@beartype
class OpenAICompatibleClient(LLMClient):
    """Wrapper for OpenAI-compatible API clients"""

    def __init__(
        self, client: Any, default_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize with an OpenAI-compatible client

        Args:
            client: OpenAI-compatible client instance (e.g., openai.OpenAI)
            default_kwargs: Kwargs merged into every ``chat.completions.create``
                call (e.g. SURF/Willma ``extra_body``).
        """
        self.client = client
        self.default_kwargs = dict(default_kwargs or {})

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a chat completion using OpenAI-compatible API

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to the API client

        Returns:
            Response dict compatible with OpenAI API format
        """
        request_kwargs = _merge_request_kwargs(self.default_kwargs, kwargs)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **request_kwargs,
        )

        message = response.choices[0].message
        content = message.content if message.content is not None else ""

        # Convert response to dict format
        return {
            "choices": [
                {
                    "message": {
                        "role": message.role,
                        "content": content,
                    }
                }
            ],
            "usage": (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if hasattr(response, "usage") and response.usage
                else None
            ),
        }
