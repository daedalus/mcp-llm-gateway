"""Tests for core domain models."""

import os
from unittest.mock import patch

from mcp_llm_gateway.core.models import (
    CompletionRequest,
    GatewayConfig,
    Model,
    Provider,
)


class TestProvider:
    """Tests for the Provider dataclass."""

    def test_provider_creation(self):
        """Test creating a provider with default values."""
        provider = Provider(
            id="openai",
            name="OpenAI",
            type="openai",
            base_url="https://api.openai.com/v1",
        )
        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert provider.type == "openai"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key is None
        assert provider.default_model == ""
        assert provider.timeout == 60
        assert provider.enabled is True

    def test_provider_creation_with_all_fields(self):
        """Test creating a provider with all fields specified."""
        provider = Provider(
            id="anthropic",
            name="Anthropic",
            type="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key="sk-ant-key",
            default_model="claude-3",
            timeout=120,
            enabled=False,
        )
        assert provider.id == "anthropic"
        assert provider.api_key == "sk-ant-key"
        assert provider.timeout == 120
        assert provider.enabled is False

    def test_provider_to_dict(self):
        """Test converting provider to dictionary."""
        provider = Provider(
            id="openai",
            name="OpenAI",
            type="openai",
            base_url="https://api.openai.com/v1",
            api_key="secret-key",
            default_model="gpt-4",
        )
        result = provider.to_dict()
        assert result["id"] == "openai"
        assert result["name"] == "OpenAI"
        assert result["has_api_key"] is True
        assert result["default_model"] == "gpt-4"

    def test_provider_to_dict_no_api_key(self):
        """Test provider to dict when no API key is set."""
        provider = Provider(
            id="openai",
            name="OpenAI",
            type="openai",
            base_url="https://api.openai.com/v1",
        )
        result = provider.to_dict()
        assert result["has_api_key"] is False


class TestModel:
    """Tests for the Model dataclass."""

    def test_model_creation(self):
        """Test creating a model with default values."""
        model = Model(id="gpt-4")
        assert model.id == "gpt-4"
        assert model.object == "model"
        assert model.created == 0
        assert model.owned_by == "unknown"
        assert model.provider_id == ""

    def test_model_creation_with_provider(self):
        """Test creating a model with provider ID."""
        model = Model(id="gpt-4", provider_id="openai")
        assert model.id == "gpt-4"
        assert model.provider_id == "openai"

    def test_model_to_dict(self):
        """Test converting model to dictionary."""
        model = Model(
            id="gpt-4", owned_by="openai", created=1234567890, provider_id="openai"
        )
        result = model.to_dict()
        assert result == {
            "id": "gpt-4",
            "object": "model",
            "created": 1234567890,
            "owned_by": "openai",
            "provider_id": "openai",
        }

    def test_model_from_dict(self):
        """Test creating model from dictionary."""
        data = {
            "id": "gpt-4",
            "object": "model",
            "created": 1234567890,
            "owned_by": "openai",
        }
        model = Model.from_dict(data, provider_id="openai")
        assert model.id == "gpt-4"
        assert model.owned_by == "openai"
        assert model.provider_id == "openai"


class TestCompletionRequest:
    """Tests for the CompletionRequest dataclass."""

    def test_completion_request_creation(self):
        """Test creating a completion request."""
        request = CompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert request.model == "gpt-4"
        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.max_tokens is None
        assert request.temperature is None

    def test_completion_request_to_dict(self):
        """Test converting completion request to dictionary."""
        request = CompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.7,
        )
        result = request.to_dict()
        assert result == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }


class TestGatewayConfig:
    """Tests for the GatewayConfig dataclass."""

    def test_gateway_config_creation(self):
        """Test creating gateway configuration."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                )
            ],
            model_list_url="https://models.dev/api/v1/models",
            cache_ttl=300,
        )
        assert len(config.providers) == 1
        assert config.providers[0].id == "openai"
        assert config.model_list_url == "https://models.dev/api/v1/models"
        assert config.cache_ttl == 300

    def test_gateway_config_to_dict(self):
        """Test converting config to dictionary."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                    api_key="key",
                )
            ],
            model_list_url="https://models.dev/api/v1/models",
            cache_ttl=300,
        )
        result = config.to_dict()
        assert result["model_list_url"] == "https://models.dev/api/v1/models"
        assert result["cache_ttl"] == 300
        assert len(result["providers"]) == 1

    def test_get_provider_with_id(self):
        """Test getting a provider by ID."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                ),
                Provider(
                    id="anthropic",
                    name="Anthropic",
                    type="anthropic",
                    base_url="https://api.anthropic.com/v1",
                    default_model="claude-3",
                ),
            ]
        )
        provider = config.get_provider("anthropic")
        assert provider is not None
        assert provider.id == "anthropic"

    def test_get_provider_without_id(self):
        """Test getting first enabled provider when no ID specified."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                    enabled=False,
                ),
                Provider(
                    id="anthropic",
                    name="Anthropic",
                    type="anthropic",
                    base_url="https://api.anthropic.com/v1",
                    default_model="claude-3",
                    enabled=True,
                ),
            ]
        )
        provider = config.get_provider(None)
        assert provider is not None
        assert provider.id == "anthropic"

    def test_get_provider_not_found(self):
        """Test getting non-existent provider returns None."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                ),
            ]
        )
        provider = config.get_provider("unknown")
        assert provider is None

    def test_get_enabled_providers(self):
        """Test getting only enabled providers."""
        config = GatewayConfig(
            providers=[
                Provider(
                    id="openai",
                    name="OpenAI",
                    type="openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-4",
                    enabled=True,
                ),
                Provider(
                    id="disabled",
                    name="Disabled",
                    type="openai",
                    base_url="https://example.com",
                    default_model="model",
                    enabled=False,
                ),
            ]
        )
        enabled = config.get_enabled_providers()
        assert len(enabled) == 1
        assert enabled[0].id == "openai"

    def test_gateway_config_from_env_success(self):
        """Test successful config creation from environment."""
        with patch.dict(
            os.environ,
            {
                "DOWNSTREAM_URL": "https://api.example.com",
                "DEFAULT_MODEL": "gpt-4",
            },
            clear=True,
        ):
            config = GatewayConfig.from_env()
            assert config.providers[0].base_url == "https://api.example.com"
            assert config.providers[0].default_model == "gpt-4"

    def test_gateway_config_from_env_with_all_vars(self):
        """Test config from environment with all variables set."""
        with patch.dict(
            os.environ,
            {
                "DOWNSTREAM_URL": "https://api.example.com",
                "DEFAULT_MODEL": "gpt-4",
                "MODEL_LIST_URL": "https://custom.models.dev/api",
                "API_KEY": "secret-key",
                "TIMEOUT": "30",
                "CACHE_TTL": "600",
            },
            clear=True,
        ):
            config = GatewayConfig.from_env()
            assert config.providers[0].base_url == "https://api.example.com"
            assert config.providers[0].api_key == "secret-key"
            assert config.providers[0].timeout == 30
            assert config.model_list_url == "https://custom.models.dev/api"
            assert config.cache_ttl == 600
