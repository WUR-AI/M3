from __future__ import annotations

from typing import Any

import openai
from beartype import beartype

from src.config import LLM_REQUEST_TIMEOUT, SURF_API_BASE, SURF_API_KEY


@beartype
def create_surf_openai_client(*, timeout: float | None = None) -> Any:
    """Build an OpenAI SDK client pointed at SURF/Willma.

    Sends both Bearer auth (OpenAI SDK default) and ``X-API-KEY`` (Willma).
    """
    if not SURF_API_BASE:
        raise ValueError(
            "SURF_API_BASE not found. Set it in your .env file.\n"
            "Example: SURF_API_BASE=https://willma.surf.nl/api/v0"
        )
    if not SURF_API_KEY:
        raise ValueError("SURF_API_KEY not found. Set it in your .env file.")

    return openai.OpenAI(
        api_key=SURF_API_KEY,
        base_url=SURF_API_BASE.rstrip("/"),
        timeout=LLM_REQUEST_TIMEOUT if timeout is None else timeout,
        default_headers={"X-API-KEY": SURF_API_KEY},
    )
