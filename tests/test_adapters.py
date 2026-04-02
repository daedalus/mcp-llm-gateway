"""Tests for HTTP adapters."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from mcp_llm_gateway.adapters.http import HTTPAdapter, ModelListAdapter
from mcp_llm_gateway.core.models import CompletionRequest, GatewayConfig


class TestHTTPAdapter:
    """Tests for HTTPAdapter."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key="test-key",
            timeout=60,
        )

    @pytest.fixture
    def adapter(self, config):
        return HTTPAdapter(config)

    def test_adapter_initialization(self, adapter, config):
        """Test that adapter is initialized with correct config."""
        assert adapter._config == config

    def test_build_headers_without_api_key(self):
        """Test headers without API key."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key=None,
        )
        adapter = HTTPAdapter(config)
        headers = adapter._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_build_headers_with_api_key(self):
        """Test headers with API key."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key="secret-key",
        )
        adapter = HTTPAdapter(config)
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer secret-key"

    def test_list_models_success(self, adapter, mock_http_response):
        """Test successful model listing."""
        with patch.object(adapter._client, "get", return_value=mock_http_response):
            models = adapter.list_models()
            assert len(models) == 2
            assert models[0].id == "gpt-4"
            assert models[1].id == "gpt-3.5-turbo"

    def test_list_models_http_error(self, adapter):
        """Test model listing when HTTP error occurs."""
        with patch.object(adapter._client, "get") as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock(status_code=500)
            )
            models = adapter.list_models()
            assert len(models) == 1
            assert models[0].id == "gpt-4"

    def test_complete_success(self, adapter, mock_completion_response):
        """Test successful completion request."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_completion_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(adapter._client, "post", return_value=mock_response):
            request = CompletionRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )
            result = adapter.complete(request)
            assert result == mock_completion_response

    def test_complete_http_error(self, adapter):
        """Test completion request when HTTP error occurs."""
        with patch.object(adapter._client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error",
                request=MagicMock(),
                response=MagicMock(status_code=400),
            )
            request = CompletionRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )
            with pytest.raises(RuntimeError, match="Downstream API error"):
                adapter.complete(request)

    def test_complete_request_error(self, adapter):
        """Test completion request when network error occurs."""
        with patch.object(adapter._client, "post") as mock_post:
            mock_post.side_effect = httpx.RequestError("Network error")
            request = CompletionRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )
            with pytest.raises(RuntimeError, match="Network error"):
                adapter.complete(request)

    def test_close(self, adapter):
        """Test closing the adapter."""
        adapter.close()


class TestModelListAdapter:
    """Tests for ModelListAdapter."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
        )

    @pytest.fixture
    def adapter(self, config):
        return ModelListAdapter(config)

    def test_adapter_initialization(self, adapter, config):
        """Test that adapter is initialized correctly."""
        assert adapter._config == config

    def test_fetch_models_success(self, adapter, mock_model_list_response):
        """Test successful model fetching."""
        with patch.object(
            adapter._client, "get", return_value=mock_model_list_response
        ):
            models = adapter.fetch_models()
            assert len(models) == 2
            assert models[0].id == "gpt-4"
            assert models[1].id == "claude-3"

    def test_fetch_models_returns_list_format(self, adapter):
        """Test fetching models when response is a list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "model-1"},
            {"id": "model-2"},
        ]

        with patch.object(adapter._client, "get", return_value=mock_response):
            models = adapter.fetch_models()
            assert len(models) == 2

    def test_fetch_models_returns_dict_with_models(self, adapter):
        """Test fetching models when response is dict with models key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"id": "model-1"}, {"id": "model-2"}]
        }

        with patch.object(adapter._client, "get", return_value=mock_response):
            models = adapter.fetch_models()
            assert len(models) == 2

    def test_fetch_models_http_error(self, adapter):
        """Test model fetching when HTTP error occurs."""
        with patch.object(adapter._client, "get") as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=MagicMock(status_code=500)
            )
            models = adapter.fetch_models()
            assert models == []

    def test_close(self, adapter):
        """Test closing the adapter."""
        adapter.close()
