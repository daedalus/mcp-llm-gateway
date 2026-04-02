"""Configuration loader for config.yaml files."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from mcp_llm_gateway.core.models import GatewayConfig, Provider


class ProviderConfig(BaseModel):
    """Pydantic model for provider configuration."""

    id: str
    name: str
    type: str = "openai"
    base_url: str
    api_key: str | None = None
    default_model: str = ""
    fallback_models: list[str] = Field(default_factory=list)
    timeout: int = 60
    enabled: bool = True


class GatewayConfigModel(BaseModel):
    """Pydantic model for gateway configuration."""

    model_list_url: str = "https://models.dev/api/v1/models"
    cache_ttl: int = 300
    providers: list[ProviderConfig] = Field(default_factory=list)


def _interpolate_env_vars(value: str) -> str:
    """Interpolate environment variables in a string.

    Supports ${VAR_NAME} and $VAR_NAME syntax.
    """
    if not isinstance(value, str):
        return value

    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replace, value)


def _interpolate_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively interpolate environment variables in a dictionary."""
    result: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, str):
            result[key] = _interpolate_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _interpolate_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _interpolate_dict(item)
                if isinstance(item, dict)
                else _interpolate_env_vars(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> GatewayConfig:
    """Load gateway configuration from config.yaml file.

    Args:
        config_path: Path to config.yaml file. If None, searches in standard locations.

    Returns:
        GatewayConfig object.

    Raises:
        FileNotFoundError: If config file not found and no legacy env vars available.
        ValueError: If configuration is invalid.
    """
    search_paths = [
        config_path,
        Path.cwd() / "config.yaml",
        Path.home() / ".config" / "mcp-llm-gateway" / "config.yaml",
        Path("/etc/mcp-llm-gateway/config.yaml"),
    ]

    config_data: dict[str, Any] | None = None

    for path in search_paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            with open(path) as f:
                raw_data = yaml.safe_load(f)
                if raw_data:
                    config_data = _interpolate_dict(raw_data)
            break

    if config_data is None:
        return GatewayConfig.from_env()

    try:
        config_model = GatewayConfigModel(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    providers = [
        Provider(
            id=p.id,
            name=p.name,
            type=p.type,
            base_url=p.base_url,
            api_key=p.api_key,
            default_model=p.default_model,
            fallback_models=p.fallback_models,
            timeout=p.timeout,
            enabled=p.enabled,
        )
        for p in config_model.providers
    ]

    if not providers:
        raise ValueError("At least one provider must be configured")

    return GatewayConfig(
        providers=providers,
        model_list_url=config_model.model_list_url,
        cache_ttl=config_model.cache_ttl,
    )
