"""Business logic services for the LLM gateway."""

import time
from typing import Any

from mcp_llm_gateway.adapters.http import HTTPAdapter, ModelListAdapter
from mcp_llm_gateway.core.models import CompletionRequest, GatewayConfig, Model


class ModelService:
    """Service for managing model discovery and caching."""

    _cache_ttl = 300  # 5 minutes

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize model service."""
        self._config = config
        self._http_adapter = HTTPAdapter(config)
        self._model_list_adapter = ModelListAdapter(config)
        self._cached_models: list[Model] | None = None
        self._cache_timestamp: float = 0

    def list_models(self, force_refresh: bool = False) -> list[Model]:
        """List available models, using cache if available."""
        if force_refresh or self._is_cache_expired():
            self._fetch_models()
        return self._cached_models or [Model(id=self._config.default_model)]

    def _is_cache_expired(self) -> bool:
        """Check if the cache has expired."""
        if self._cached_models is None:
            return True
        return (time.time() - self._cache_timestamp) > self._cache_ttl

    def _fetch_models(self) -> None:
        """Fetch models from remote or fallback to downstream."""
        try:
            models = self._model_list_adapter.fetch_models()
            if not models:
                models = self._http_adapter.list_models()
        except Exception:
            models = self._http_adapter.list_models()

        self._cached_models = models
        self._cache_timestamp = time.time()

    def get_model(self, model_id: str | None) -> Model:
        """Get a specific model by ID, or return default."""
        if model_id:
            return Model(id=model_id)
        return Model(id=self._config.default_model)

    def close(self) -> None:
        """Close adapter connections."""
        self._http_adapter.close()
        self._model_list_adapter.close()


class CompletionService:
    """Service for handling completion requests."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize completion service."""
        self._config = config
        self._http_adapter = HTTPAdapter(config)

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a completion request to the downstream API."""
        model_id = model or self._config.default_model

        messages = [{"role": "user", "content": prompt}]
        request = CompletionRequest(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return self._http_adapter.complete(request)

    def close(self) -> None:
        """Close adapter connections."""
        self._http_adapter.close()


class ConfigService:
    """Service for managing gateway configuration."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize config service."""
        self._config = config

    def get_config(self) -> dict[str, Any]:
        """Get current configuration as dictionary."""
        return self._config.to_dict()
