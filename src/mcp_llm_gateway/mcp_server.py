"""MCP Server implementation using FastMCP."""

from typing import Any

import fastmcp

from mcp_llm_gateway.core.models import GatewayConfig
from mcp_llm_gateway.services.gateway import (
    CompletionService,
    ConfigService,
    ModelService,
)

mcp = fastmcp.FastMCP("mcp-llm-gateway")

_model_service: ModelService | None = None
_completion_service: CompletionService | None = None
_config_service: ConfigService | None = None


def _get_services() -> tuple[ModelService, "CompletionService", "ConfigService"]:
    """Initialize and return services (singleton pattern)."""
    global _model_service, _completion_service, _config_service

    if _model_service is None:
        config = GatewayConfig.from_env()
        _model_service = ModelService(config)
        _completion_service = CompletionService(config)
        _config_service = ConfigService(config)

    return _model_service, _completion_service, _config_service  # type: ignore[return-value]


@mcp.tool()
def list_models() -> list[dict[str, Any]]:
    """List all available models from the remote endpoint.

    Fetches models from the configured model list URL, with caching.
    Falls back to the downstream API if the remote list is unavailable.

    Returns:
        List of model objects with id, name, and metadata.

    Example:
        >>> list_models()
        [{"id": "gpt-4", "object": "model", "created": 1234567890, "owned_by": "openai"}]
    """
    model_service, _, _ = _get_services()
    models = model_service.list_models()
    return [m.to_dict() for m in models]


@mcp.tool()
def complete(
    prompt: str,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Send a completion request to the downstream LLM provider.

    Proxies the request to the configured OpenAI-compatible downstream endpoint.

    Args:
        prompt: The input prompt for the model.
        model: Optional model ID. Uses DEFAULT_MODEL if not specified.
        max_tokens: Optional maximum tokens to generate.
        temperature: Optional sampling temperature (0.0 to 2.0).

    Returns:
        Completion response from the downstream provider.

    Example:
        >>> complete("Hello, world!", model="gpt-4", max_tokens=100)
        {"id": "...", "object": "chat.completion", ...}
    """
    _, completion_service, _ = _get_services()
    return completion_service.complete(
        prompt=prompt,
        model=model,
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
        Configuration details including endpoint URLs and settings.

    Example:
        >>> config_info()
        {"downstream_url": "...", "default_model": "...", ...}
    """
    _, _, config_service = _get_services()
    return config_service.get_config()


def main() -> int:
    """Main entry point for the MCP server."""
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
