"""
Centralized configuration for the Metadata Agent.

All configurable settings should be defined here.
You can also override these via environment variables.

Supported LLM Providers:
- "google": Google Gemini models (requires GOOGLE_API_KEY)
- "surf": Custom OpenAI-compatible endpoint (requires SURF_API_BASE and optionally SURF_API_KEY)
- "openai": OpenAI models (requires OPENAI_API_KEY)
"""
import os
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

# LLM Provider: "google", "surf", "openai"
# Can be overridden by environment variable: LLM_PROVIDER
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "google": {
        "default_model": "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "description": "Google Gemini models",
    },
    "surf": {
        "default_model": "default-text-large",
        "api_key_env": "SURF_API_KEY",
        "base_url_env": "SURF_API_BASE",
        "description": "Custom OpenAI-compatible endpoint (e.g., vLLM, TGI)",
    },
    "openai": {
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI models",
    },
}

# =============================================================================
# LLM MODEL CONFIGURATION
# =============================================================================

# Default model - uses provider's default if not specified
# Can be overridden by environment variable: LLM_MODEL
DEFAULT_MODEL = os.getenv("LLM_MODEL", None)  # None means use provider default

# Default temperature for planning (lower = more deterministic)
# Can be overridden by environment variable: LLM_TEMPERATURE_PLANNING
PLANNING_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE_PLANNING", "0.0"))

# Default temperature for players (higher = more creative)
# Can be overridden by environment variable: LLM_TEMPERATURE_PLAYER  
PLAYER_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE_PLAYER", "0.3"))

# Max model↔tool rounds per player task.
PLAYER_MAX_TOOL_ITERATIONS = int(os.getenv("PLAYER_MAX_TOOL_ITERATIONS", "16"))

# Per-request LLM timeout (seconds). Prevents calls from hanging forever.
# Can be overridden by environment variable: LLM_REQUEST_TIMEOUT
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))


# =============================================================================
# PROVIDER-SPECIFIC API KEYS AND ENDPOINTS
# =============================================================================

# Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Surf (custom OpenAI-compatible endpoint)
SURF_API_BASE = os.getenv("SURF_API_BASE")  # e.g., "http://localhost:8000/v1"
SURF_API_KEY = os.getenv("SURF_API_KEY")  # Required for Surf provider
SURF_ENABLE_THINKING = os.getenv("SURF_ENABLE_THINKING", "false").lower() == "true"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =============================================================================
# EXECUTION DEFAULTS
# =============================================================================

# Default execution topology
# Can be overridden by environment variable: DEFAULT_TOPOLOGY
DEFAULT_TOPOLOGY = os.getenv("DEFAULT_TOPOLOGY", "single")

# Default metadata standard
# Can be overridden by environment variable: DEFAULT_METADATA_STANDARD
DEFAULT_METADATA_STANDARD = os.getenv("DEFAULT_METADATA_STANDARD", "basic")

# TUI logging level (e.g., DEBUG, INFO, WARNING, ERROR)
# Can be overridden by environment variable: TUI_LOG_LEVEL
TUI_LOG_LEVEL = os.getenv("TUI_LOG_LEVEL", "INFO").upper()

# Comma-separated logger prefixes to suppress from TUI output
# Example: "src.orchestrator.step_executor,src.players.player"
TUI_LOG_SUPPRESSED_LOGGERS = os.getenv(
    "TUI_LOG_SUPPRESSED_LOGGERS",
    "src.orchestrator.step_executor,src.players.player",
)

# TUI user-facing log verbosity:
# - "quiet": status line only (no streamed log lines in chat)
# - "normal": status line + explicit UI messages
# - "debug": status line + all streamed log lines
TUI_UI_VERBOSITY = os.getenv("TUI_UI_VERBOSITY", "normal").lower()


# =============================================================================
# LLM FACTORY
# =============================================================================

def get_model_name(override: Optional[str] = None) -> str:
    """
    Get the model name to use.
    
    Priority:
    1. Override parameter
    2. LLM_MODEL environment variable
    3. Provider's default model
    """
    if override:
        return override
    if DEFAULT_MODEL:
        return DEFAULT_MODEL
    return PROVIDER_CONFIGS.get(LLM_PROVIDER, {}).get("default_model", "gpt-4o-mini")


def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Factory function to create an LLM instance based on the configured provider.
    
    Args:
        model_name: Model name (uses default if not specified)
        temperature: LLM temperature
        provider: Override the default provider
        **kwargs: Additional arguments passed to the LLM constructor
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If provider is not supported or required config is missing
    """
    provider = provider or LLM_PROVIDER
    model = get_model_name(model_name)
    
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file."
            )
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            **kwargs
        )
    
    elif provider == "surf":
        from langchain_openai import ChatOpenAI
        
        if not SURF_API_BASE:
            raise ValueError(
                "SURF_API_BASE not found. Set it in your .env file.\n"
                "Example: SURF_API_BASE=http://localhost:8000/v1"
            )
        
        if not SURF_API_KEY:
            raise ValueError(
                "SURF_API_KEY not found. Set it in your .env file."
            )
        
        model_kwargs = kwargs.pop("model_kwargs", {}) or {}
        extra_body = dict(model_kwargs.get("extra_body") or {})
        # Mistral tokenizers reject chat_template_kwargs on Willma/SURF.
        if "mistral" not in model.lower() and "devstral" not in model.lower():
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            # Disable reasoning/thinking output by default for Surf-compatible backends.
            # This helps keep responses clean for downstream structured parsing.
            chat_template_kwargs.setdefault("enable_thinking", SURF_ENABLE_THINKING)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if extra_body:
            model_kwargs["extra_body"] = extra_body

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=SURF_API_KEY,
            openai_api_base=SURF_API_BASE,
            model_kwargs=model_kwargs,
            request_timeout=kwargs.pop("request_timeout", LLM_REQUEST_TIMEOUT),
            **kwargs
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in your .env file."
            )
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY,
            request_timeout=kwargs.pop("request_timeout", LLM_REQUEST_TIMEOUT),
            **kwargs
        )
    
    else:
        available = list(PROVIDER_CONFIGS.keys())
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. Available: {available}"
        )


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def get_config_summary() -> str:
    """Return a summary of current configuration."""
    provider_config = PROVIDER_CONFIGS.get(LLM_PROVIDER, {})
    model = get_model_name()
    
    # Check API key status
    api_key_env = provider_config.get("api_key_env", "")
    api_key_set = bool(os.getenv(api_key_env)) if api_key_env else False
    
    return f"""
Configuration Summary:
----------------------
LLM Provider: {LLM_PROVIDER} ({provider_config.get('description', 'Unknown')})
LLM Model: {model}
Planning Temperature: {PLANNING_TEMPERATURE}
Player Temperature: {PLAYER_TEMPERATURE}
Default Topology: {DEFAULT_TOPOLOGY}
Default Metadata Standard: {DEFAULT_METADATA_STANDARD}
TUI Log Level: {TUI_LOG_LEVEL}
API Key ({api_key_env}): {'Set' if api_key_set else 'Not Set'}
"""
