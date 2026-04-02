"""HTTP adapter for downstream API interactions."""

from typing import Any

import httpx

from mcp_llm_gateway.core.models import CompletionRequest, GatewayConfig, Model


class HTTPAdapter:
    """HTTP client for downstream API communication."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize HTTP adapter with configuration."""
        self._config = config
        self._client = httpx.Client(
            base_url=config.downstream_url,
            timeout=config.timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def list_models(self) -> list[Model]:
        """Fetch available models from downstream endpoint."""
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [Model.from_dict(m) for m in data.get("data", [])]
        except httpx.HTTPError:
            return [Model(id=self._config.default_model)]

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

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize model list adapter."""
        self._config = config
        self._client = httpx.Client(timeout=30)

    def fetch_models(self) -> list[Model]:
        """Fetch models from the remote model list endpoint."""
        try:
            response = self._client.get(self._config.model_list_url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return [Model.from_dict(m) for m in data]
            if isinstance(data, dict) and "models" in data:
                return [Model.from_dict(m) for m in data.get("models", [])]
            return []
        except httpx.HTTPError:
            return []

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
