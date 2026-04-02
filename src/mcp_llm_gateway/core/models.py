"""Core domain models for the LLM gateway."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Provider:
    """Represents a configured LLM provider."""

    id: str
    name: str
    type: str
    base_url: str
    api_key: str | None = None
    default_model: str = ""
    fallback_models: list[str] = field(default_factory=list)
    timeout: int = 60
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "fallback_models": self.fallback_models,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "has_api_key": self.api_key is not None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provider":
        """Create Provider from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=data.get("type", "openai"),
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key"),
            default_model=data.get("default_model", ""),
            fallback_models=data.get("fallback_models", []),
            timeout=data.get("timeout", 60),
            enabled=data.get("enabled", True),
        )


@dataclass
class Model:
    """Represents an available LLM model."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "unknown"
    provider_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
            "provider_id": self.provider_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], provider_id: str = "") -> "Model":
        """Create Model from dictionary."""
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", "unknown"),
            provider_id=provider_id,
        )


@dataclass
class CompletionRequest:
    """Request payload for completions."""

    model: str
    messages: list[dict[str, str]]
    max_tokens: int | None = None
    temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for downstream API."""
        result: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
        }
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        return result


@dataclass
class GatewayConfig:
    """Gateway configuration."""

    providers: list[Provider]
    model_list_url: str = "https://models.dev/api/v1/models"
    cache_ttl: int = 300

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Create configuration from environment variables (legacy fallback)."""
        import os

        downstream_url = os.environ.get("DOWNSTREAM_URL")
        if not downstream_url:
            raise ValueError("DOWNSTREAM_URL environment variable is required")

        default_model = os.environ.get("DEFAULT_MODEL")
        if not default_model:
            raise ValueError("DEFAULT_MODEL environment variable is required")

        return cls(
            providers=[
                Provider(
                    id="default",
                    name="Default Provider",
                    type="openai",
                    base_url=downstream_url,
                    api_key=os.environ.get("API_KEY"),
                    default_model=default_model,
                    timeout=int(os.environ.get("TIMEOUT", "60")),
                )
            ],
            model_list_url=os.environ.get(
                "MODEL_LIST_URL", "https://models.dev/api/v1/models"
            ),
            cache_ttl=int(os.environ.get("CACHE_TTL", "300")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for resource response."""
        return {
            "model_list_url": self.model_list_url,
            "cache_ttl": self.cache_ttl,
            "providers": [p.to_dict() for p in self.providers],
        }

    def get_provider(self, provider_id: str | None = None) -> Provider | None:
        """Get a provider by ID, or return the first enabled provider."""
        if provider_id:
            for p in self.providers:
                if p.id == provider_id:
                    return p
            return None
        for p in self.providers:
            if p.enabled:
                return p
        return None

    def get_enabled_providers(self) -> list[Provider]:
        """Get list of enabled providers."""
        return [p for p in self.providers if p.enabled]
