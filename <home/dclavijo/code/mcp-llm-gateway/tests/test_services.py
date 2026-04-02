"""Tests for gateway services."""

import pytest
from unittest.mock import MagicMock, patch
import time

from mcp_llm_gateway.services.gateway import (
    CompletionService,
    ConfigService,
    ModelService,
)
from mcp_llm_gateway.core.models import GatewayConfig, Model


class TestModelService:
    """Tests for ModelService."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
        )

    @pytest.fixture
    def service(self, config):
        with (
            patch("mcp_llm_gateway.services.gateway.HTTPAdapter") as mock_http,
            patch(
                "mcp_llm_gateway.services.gateway.ModelListAdapter"
            ) as mock_model_list,
        ):
            mock_http_instance = MagicMock()
            mock_model_list_instance = MagicMock()
            mock_http.return_value = mock_http_instance
            mock_model_list.return_value = mock_model_list_instance

            mock_http_instance.list_models.return_value = [
                Model(id="gpt-4"),
                Model(id="gpt-3.5-turbo"),
            ]
            mock_model_list_instance.fetch_models.return_value = [
                Model(id="claude-3"),
                Model(id="gemini-pro"),
            ]

            service = ModelService(config)
            yield service
            service.close()

    def test_list_models(self, service):
        """Test listing models."""
        models = service.list_models()
        assert len(models) == 2

    def test_list_models_force_refresh(self, service):
        """Test force refresh of model list."""
        models = service.list_models(force_refresh=True)
        assert len(models) == 2

    def test_list_models_fallback_on_empty(self):
        """Test fallback to default model when list is empty."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
        )

        with (
            patch("mcp_llm_gateway.services.gateway.HTTPAdapter") as mock_http,
            patch(
                "mcp_llm_gateway.services.gateway.ModelListAdapter"
            ) as mock_model_list,
        ):
            mock_http_instance = MagicMock()
            mock_model_list_instance = MagicMock()
            mock_http.return_value = mock_http_instance
            mock_model_list.return_value = mock_model_list_instance

            mock_model_list_instance.fetch_models.return_value = []
            mock_http_instance.list_models.return_value = []

            service = ModelService(config)
            models = service.list_models()
            assert len(models) == 1
            assert models[0].id == "gpt-4"
            service.close()

    def test_get_model_with_id(self, service):
        """Test getting a specific model by ID."""
        model = service.get_model("gpt-4")
        assert model.id == "gpt-4"

    def test_get_model_without_id(self, service):
        """Test getting default model when no ID provided."""
        model = service.get_model(None)
        assert model.id == "gpt-4"

    def test_close(self, service):
        """Test closing the service."""
        service.close()


class TestCompletionService:
    """Tests for CompletionService."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
        )

    @pytest.fixture
    def service(self, config):
        with patch("mcp_llm_gateway.services.gateway.HTTPAdapter") as mock_http:
            mock_http_instance = MagicMock()
            mock_http.return_value = mock_http_instance

            mock_http_instance.complete.return_value = {
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "Hello!"}}],
            }

            service = CompletionService(config)
            yield service
            service.close()

    def test_complete_with_defaults(self, service):
        """Test completion with default model."""
        result = service.complete("Hello")
        assert "id" in result

    def test_complete_with_custom_model(self, service):
        """Test completion with custom model."""
        result = service.complete("Hello", model="gpt-3.5-turbo")
        assert "id" in result

    def test_complete_with_options(self, service):
        """Test completion with all options."""
        result = service.complete(
            "Hello",
            model="gpt-4",
            max_tokens=100,
            temperature=0.7,
        )
        assert "id" in result

    def test_close(self, service):
        """Test closing the service."""
        service.close()


class TestConfigService:
    """Tests for ConfigService."""

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
    def service(self, config):
        return ConfigService(config)

    def test_get_config(self, service):
        """Test getting configuration."""
        config = service.get_config()
        assert config["downstream_url"] == "https://api.example.com"
        assert config["default_model"] == "gpt-4"
        assert config["has_api_key"] is True
        assert config["timeout"] == 60
