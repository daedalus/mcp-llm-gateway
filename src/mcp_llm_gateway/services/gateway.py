"""Business logic services for the LLM gateway."""

import time
from typing import Any

from mcp_llm_gateway.adapters.http import HTTPAdapter, ModelListAdapter
from mcp_llm_gateway.core.logging import GatewayLogger
from mcp_llm_gateway.core.models import (
    CompletionRequest,
    GatewayConfig,
    Model,
    Provider,
)


class ModelService:
    """Service for managing model discovery and caching."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize model service."""
        self._config = config
        self._logger = GatewayLogger("ModelService")
        self._model_list_adapter = ModelListAdapter(config.model_list_url)
        self._http_adapters: dict[str, HTTPAdapter] = {}
        self._cached_models: list[Model] | None = None
        self._cache_timestamp: float = 0
        self._available_models: dict[str, list[str]] = {}

    def _get_http_adapter(self, provider: Provider) -> HTTPAdapter:
        """Get or create HTTP adapter for a provider."""
        if provider.id not in self._http_adapters:
            self._http_adapters[provider.id] = HTTPAdapter(provider)
        return self._http_adapters[provider.id]

    def list_models(
        self,
        provider: str | None = None,
        force_refresh: bool = False,
    ) -> list[Model]:
        """List available models from specified provider or all providers."""
        if force_refresh or self._is_cache_expired():
            self._fetch_models(provider)

        if provider:
            return [m for m in (self._cached_models or []) if m.provider_id == provider]
        return self._cached_models or []

    def _is_cache_expired(self) -> bool:
        """Check if the cache has expired."""
        if self._cached_models is None:
            return True
        return (time.time() - self._cache_timestamp) > self._config.cache_ttl

    def _fetch_models(self, provider_filter: str | None = None) -> None:
        """Fetch models from remote or fallback to provider endpoints."""
        all_models: list[Model] = []

        enabled_providers = self._config.get_enabled_providers()

        if provider_filter:
            enabled_providers = [
                p for p in enabled_providers if p.id == provider_filter
            ]

        for provider in enabled_providers:
            try:
                http_adapter = self._get_http_adapter(provider)
                models = http_adapter.list_models()
                if models:
                    all_models.extend(models)
                    self._available_models[provider.id] = [m.id for m in models]
                else:
                    all_models.append(
                        Model(
                            id=provider.default_model,
                            provider_id=provider.id,
                            owned_by=provider.name,
                        )
                    )
            except Exception as e:
                self._logger.error(
                    f"Failed to fetch models from provider: {provider.id} | error={e}"
                )
                all_models.append(
                    Model(
                        id=provider.default_model,
                        provider_id=provider.id,
                        owned_by=provider.name,
                    )
                )

        if not all_models:
            remote_models = self._model_list_adapter.fetch_models()
            if remote_models:
                all_models = remote_models

        self._cached_models = all_models
        self._cache_timestamp = time.time()

    def get_model(self, model_id: str | None, provider_id: str | None = None) -> Model:
        """Get a specific model by ID, or return default for provider."""
        provider = self._config.get_provider(provider_id)
        if model_id:
            return Model(id=model_id, provider_id=provider.id if provider else "")
        if provider:
            return Model(id=provider.default_model, provider_id=provider.id)
        return Model(id="")

    def get_available_models(self, provider_id: str) -> list[str]:
        """Get list of available models for a provider from models.dev."""
        return self._available_models.get(provider_id, [])

    def close(self) -> None:
        """Close adapter connections."""
        for adapter in self._http_adapters.values():
            adapter.close()
        self._model_list_adapter.close()


class CompletionService:
    """Service for handling completion requests."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize completion service."""
        self._config = config
        self._logger = GatewayLogger("CompletionService")
        self._http_adapters: dict[str, HTTPAdapter] = {}

    def _get_http_adapter(self, provider: Provider) -> HTTPAdapter:
        """Get or create HTTP adapter for a provider."""
        if provider.id not in self._http_adapters:
            self._http_adapters[provider.id] = HTTPAdapter(provider)
        return self._http_adapters[provider.id]

    def _try_complete(
        self,
        http_adapter: HTTPAdapter,
        provider: Provider,
        request: CompletionRequest,
        model_id: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Attempt a completion request.

        Returns:
            Tuple of (response, error). If error is None, request succeeded.
        """
        start_time = time.time()
        try:
            response = http_adapter.complete(request)
            duration_ms = (time.time() - start_time) * 1000
            self._logger.log_request(
                provider=provider.id,
                model=model_id,
                prompt=request.messages[0].get("content", "")[:50]
                if request.messages
                else "",
                success=True,
                duration_ms=duration_ms,
            )
            return response, None
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            self._logger.log_request(
                provider=provider.id,
                model=model_id,
                prompt=request.messages[0].get("content", "")[:50]
                if request.messages
                else "",
                success=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
            return None, error_msg

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a completion request to the downstream API with fallback support."""
        selected_provider = self._config.get_provider(provider)
        if not selected_provider:
            available = [p.id for p in self._config.get_enabled_providers()]
            raise ValueError(f"Provider not found: {provider}. Available: {available}")

        if not selected_provider.enabled:
            raise ValueError(f"Provider is disabled: {selected_provider.id}")

        model_id = model or selected_provider.default_model
        messages = [{"role": "user", "content": prompt}]

        models_to_try = [model_id]
        if not model:
            models_to_try = [selected_provider.default_model] + list(
                selected_provider.fallback_models
            )

        http_adapter = self._get_http_adapter(selected_provider)

        last_error: str | None = None
        for attempt_model in models_to_try:
            request = CompletionRequest(
                model=attempt_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if attempt_model != model_id:
                self._logger.log_model_fallback(
                    provider=selected_provider.id,
                    failed_model=model_id,
                    fallback_model=attempt_model,
                )

            response, error = self._try_complete(
                http_adapter, selected_provider, request, attempt_model
            )

            if response is not None:
                return response

            last_error = error

        raise RuntimeError(
            f"All models failed for provider {selected_provider.id}. "
            f"Models tried: {models_to_try}. Last error: {last_error}"
        )

    def close(self) -> None:
        """Close adapter connections."""
        for adapter in self._http_adapters.values():
            adapter.close()


class ConfigService:
    """Service for managing gateway configuration."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize config service."""
        self._config = config

    def get_config(self) -> dict[str, Any]:
        """Get current configuration as dictionary."""
        return self._config.to_dict()

    def get_providers(self) -> list[dict[str, Any]]:
        """Get list of providers as dictionaries."""
        return [p.to_dict() for p in self._config.providers]

    def get_enabled_providers(self) -> list[dict[str, Any]]:
        """Get list of enabled providers as dictionaries."""
        return [p.to_dict() for p in self._config.get_enabled_providers()]
