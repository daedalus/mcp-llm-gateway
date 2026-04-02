"""HTTP adapter for downstream API interactions."""

from typing import Any

import httpx

from mcp_llm_gateway.core.models import CompletionRequest, Model, Provider


class HTTPAdapter:
    """HTTP client for downstream API communication."""

    def __init__(self, provider: Provider) -> None:
        """Initialize HTTP adapter with provider configuration."""
        self._provider = provider
        self._client = httpx.Client(
            base_url=provider.base_url,
            timeout=provider.timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._provider.api_key:
            headers["Authorization"] = f"Bearer {self._provider.api_key}"
        return headers

    def list_models(self) -> list[Model]:
        """Fetch available models from downstream endpoint."""
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [Model.from_dict(m, self._provider.id) for m in data.get("data", [])]
        except httpx.HTTPError:
            return [
                Model(id=self._provider.default_model, provider_id=self._provider.id)
            ]

    def complete(self, request: CompletionRequest) -> dict[str, Any]:
        """Send completion request to downstream API."""
        try:
            response = self._client.post(
                "/v1/chat/completions",
                json=request.to_dict(),
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Downstream API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error: {e}")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class ModelListAdapter:
    """Adapter for fetching model list from remote endpoints."""

    def __init__(self, model_list_url: str) -> None:
        """Initialize model list adapter."""
        self._model_list_url = model_list_url
        self._client = httpx.Client(timeout=30)

    def fetch_models(self) -> list[Model]:
        """Fetch models from the remote model list endpoint (models.dev)."""
        try:
            response = self._client.get(self._model_list_url)
            response.raise_for_status()
            data = response.json()

            models: list[Model] = []

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("modelId") or item.get("id")
                        if model_id:
                            models.append(
                                Model(
                                    id=model_id,
                                    owned_by=item.get("providerId", "unknown"),
                                )
                            )

            elif isinstance(data, dict):
                if "models" in data:
                    for provider_models in data.get("models", {}).values():
                        if isinstance(provider_models, list):
                            for m in provider_models:
                                model_id = m.get("modelId") or m.get("id")
                                if model_id:
                                    models.append(
                                        Model(
                                            id=model_id,
                                            owned_by=m.get("providerId", "unknown"),
                                        )
                                    )

                elif "providers" in data:
                    for provider in data.get("providers", []):
                        provider_id = provider.get("id", "")
                        for model in provider.get("models", []):
                            model_id = model.get("modelId") or model.get("id")
                            if model_id:
                                models.append(
                                    Model(
                                        id=model_id,
                                        owned_by=provider_id,
                                    )
                                )

            return models
        except httpx.HTTPError:
            return []
        except Exception:
            return []

    def get_models_by_provider(self, provider_id: str) -> list[str]:
        """Get list of model IDs for a specific provider."""
        models = self.fetch_models()
        return [m.id for m in models if m.owned_by == provider_id]

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
