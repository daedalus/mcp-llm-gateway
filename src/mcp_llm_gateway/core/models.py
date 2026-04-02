"""Core domain models for the LLM gateway."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Model:
    """Represents an available LLM model."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        """Create Model from dictionary."""
        return cls(
            id=data.get("id", ""),
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", "unknown"),
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

    downstream_url: str
    model_list_url: str
    default_model: str
    api_key: str | None = None
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Create configuration from environment variables."""
        import os

        downstream_url = os.environ.get("DOWNSTREAM_URL")
        if not downstream_url:
            raise ValueError("DOWNSTREAM_URL environment variable is required")

        default_model = os.environ.get("DEFAULT_MODEL")
        if not default_model:
            raise ValueError("DEFAULT_MODEL environment variable is required")

        return cls(
            downstream_url=downstream_url,
            model_list_url=os.environ.get(
                "MODEL_LIST_URL", "https://models.dev/api/v1/models"
            ),
            default_model=default_model,
            api_key=os.environ.get("API_KEY"),
            timeout=int(os.environ.get("TIMEOUT", "60")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for resource response."""
        return {
            "downstream_url": self.downstream_url,
            "model_list_url": self.model_list_url,
            "default_model": self.default_model,
            "timeout": self.timeout,
            "has_api_key": self.api_key is not None,
        }
