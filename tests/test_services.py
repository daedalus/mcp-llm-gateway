"""Tests for gateway services."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_gateway.core.models import GatewayConfig, Model, Provider
from mcp_llm_gateway.services.gateway import (
    CompletionService,
    ConfigService,
    ModelService,
)


class TestModelService:
    """Tests for ModelService."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.example.com",
                    default_model="gpt-4",
                ),
                Provider(
                    id="anthropic",
                    name="Anthropic",
                    type="anthropic",
                    base_url="https://api.anthropic.com",
                    default_model="claude-3",
                ),
            ],
            model_list_url="https://models.dev/api/v1/models",
            cache_ttl=300,
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
                Model(id="gpt-4", provider_id="openai"),
                Model(id="gpt-3.5-turbo", provider_id="openai"),
            ]
            mock_model_list_instance.fetch_models.return_value = []

            service = ModelService(config)
            yield service
            service.close()

    def test_list_models(self, service):
        """Test listing models."""
        models = service.list_models(provider="openai")
        assert len(models) == 2

    def test_list_models_force_refresh(self, service):
        """Test force refresh of model list."""
        models = service.list_models(provider="openai", force_refresh=True)
        assert len(models) == 2

    def test_list_models_filter_by_provider(self, service):
        """Test filtering models by provider."""
        models = service.list_models(provider="openai")
        assert len(models) == 2

    def test_get_model_with_id(self, service):
        """Test getting a specific model by ID."""
        model = service.get_model("gpt-4", provider_id="openai")
        assert model.id == "gpt-4"
        assert model.provider_id == "openai"

    def test_get_model_without_id(self, service):
        """Test getting default model when no ID provided."""
        model = service.get_model(None, provider_id="openai")
        assert model.id == "gpt-4"
        assert model.provider_id == "openai"

    def test_close(self, service):
        """Test closing the service."""
        service.close()


class TestCompletionService:
    """Tests for CompletionService."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.example.com",
                    default_model="gpt-4",
                ),
            ],
            model_list_url="https://models.dev/api/v1/models",
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

    def test_complete_with_provider(self, service):
        """Test completion with provider specified."""
        result = service.complete("Hello", provider="openai")
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

    def test_complete_provider_not_found(self, service):
        """Test completion with unknown provider raises error."""
        with pytest.raises(ValueError, match="Provider not found"):
            service.complete("Hello", provider="unknown")

    def test_close(self, service):
        """Test closing the service."""
        service.close()


class TestConfigService:
    """Tests for ConfigService."""

    @pytest.fixture
    def config(self):
        return GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.example.com",
                    default_model="gpt-4",
                    api_key="test-key",
                    enabled=True,
                ),
                Provider(
                    id="anthropic",
                    name="Anthropic",
                    type="anthropic",
                    base_url="https://api.anthropic.com",
                    default_model="claude-3",
                    enabled=False,
                ),
            ],
            model_list_url="https://models.dev/api/v1/models",
            cache_ttl=300,
        )

    @pytest.fixture
    def service(self, config):
        return ConfigService(config)

    def test_get_config(self, service):
        """Test getting configuration."""
        config = service.get_config()
        assert config["model_list_url"] == "https://models.dev/api/v1/models"
        assert config["cache_ttl"] == 300
        assert len(config["providers"]) == 2

    def test_get_providers(self, service):
        """Test getting all providers."""
        providers = service.get_providers()
        assert len(providers) == 2
        assert providers[0]["id"] == "openai"
        assert providers[1]["id"] == "anthropic"

    def test_get_enabled_providers(self, service):
        """Test getting only enabled providers."""
        enabled = service.get_enabled_providers()
        assert len(enabled) == 1
        assert enabled[0]["id"] == "openai"
