"""MCP Server implementation using FastMCP."""

import os
from typing import Any

import fastmcp

from mcp_llm_gateway.core import load_config, setup_logging
from mcp_llm_gateway.services.gateway import (
    CompletionService,
    ConfigService,
    ModelService,
)

setup_logging(
    log_file=os.environ.get("LOG_FILE"),
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
)

mcp = fastmcp.FastMCP("mcp-llm-gateway")

_model_service: ModelService | None = None
_completion_service: CompletionService | None = None
_config_service: ConfigService | None = None


def _get_services() -> tuple[ModelService, "CompletionService", "ConfigService"]:
    """Initialize and return services (singleton pattern)."""
    global _model_service, _completion_service, _config_service

    if _model_service is None:
        config = load_config()
        _model_service = ModelService(config)
        _completion_service = CompletionService(config)
        _config_service = ConfigService(config)

    return _model_service, _completion_service, _config_service  # type: ignore[return-value]


@mcp.tool()
def list_models(provider: str | None = None) -> list[dict[str, Any]]:
    """List all available models from the configured providers.

    Fetches models from the configured providers, with caching.
    Can filter by provider ID.

    Args:
        provider: Optional provider ID to filter models.

    Returns:
        List of model objects with id, name, and metadata.

    Example:
        >>> list_models()
        [{"id": "gpt-4", "object": "model", "created": 1234567890, "owned_by": "openai", "provider_id": "openai"}]
        >>> list_models(provider="anthropic")
        [{"id": "claude-3", "object": "model", "provider_id": "anthropic"}]
    """
    model_service, _, _ = _get_services()
    models = model_service.list_models(provider=provider)
    return [m.to_dict() for m in models]


@mcp.tool()
def complete(
    prompt: str,
    model: str | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Send a completion request to the downstream LLM provider.

    Proxies the request to the configured OpenAI-compatible downstream endpoint.

    Args:
        prompt: The input prompt for the model.
        model: Optional model ID. Uses provider default if not specified.
        provider: Optional provider ID. Uses first enabled provider if not specified.
        max_tokens: Optional maximum tokens to generate.
        temperature: Optional sampling temperature (0.0 to 2.0).

    Returns:
        Completion response from the downstream provider.

    Example:
        >>> complete("Hello, world!", model="gpt-4", max_tokens=100)
        {"id": "...", "object": "chat.completion", ...}
        >>> complete("Hello", provider="anthropic", model="claude-3")
        {"id": "...", "object": "chat.completion", ...}
    """
    _, completion_service, _ = _get_services()
    return completion_service.complete(
        prompt=prompt,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@mcp.resource("models://list")
def models_list() -> list[dict[str, Any]]:
    """Resource URI that returns the list of available models.

    Returns:
        JSON array of model objects.

    Example:
        >>> models_list()
        [{"id": "gpt-4", "object": "model", ...}]
    """
    model_service, _, _ = _get_services()
    models = model_service.list_models()
    return [m.to_dict() for m in models]


@mcp.resource("config://info")
def config_info() -> dict[str, Any]:
    """Resource URI that returns current gateway configuration.

    Returns:
        Configuration details including providers, model list URL, etc.

    Example:
        >>> config_info()
        {"model_list_url": "...", "cache_ttl": 300, "providers": [...]}
    """
    _, _, config_service = _get_services()
    return config_service.get_config()


@mcp.resource("providers://list")
def providers_list() -> list[dict[str, Any]]:
    """Resource URI that returns the list of configured providers.

    Returns:
        JSON array of provider objects.

    Example:
        >>> providers_list()
        [{"id": "openai", "name": "OpenAI", "type": "openai", ...}]
    """
    _, _, config_service = _get_services()
    return config_service.get_providers()


def main() -> int:
    """Main entry point for the MCP server."""
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
