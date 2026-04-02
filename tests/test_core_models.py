"""Tests for core domain models."""

import os
from unittest.mock import patch

import pytest

from mcp_llm_gateway.core.models import CompletionRequest, GatewayConfig, Model


class TestModel:
    """Tests for the Model dataclass."""

    def test_model_creation(self):
        """Test creating a model with default values."""
        model = Model(id="gpt-4")
        assert model.id == "gpt-4"
        assert model.object == "model"
        assert model.created == 0
        assert model.owned_by == "unknown"

    def test_model_creation_with_all_fields(self):
        """Test creating a model with all fields specified."""
        model = Model(
            id="gpt-4",
            object="model",
            created=1234567890,
            owned_by="openai",
        )
        assert model.id == "gpt-4"
        assert model.object == "model"
        assert model.created == 1234567890
        assert model.owned_by == "openai"

    def test_model_to_dict(self):
        """Test converting model to dictionary."""
        model = Model(id="gpt-4", owned_by="openai", created=1234567890)
        result = model.to_dict()
        assert result == {
            "id": "gpt-4",
            "object": "model",
            "created": 1234567890,
            "owned_by": "openai",
        }

    def test_model_from_dict(self):
        """Test creating model from dictionary."""
        data = {
            "id": "gpt-4",
            "object": "model",
            "created": 1234567890,
            "owned_by": "openai",
        }
        model = Model.from_dict(data)
        assert model.id == "gpt-4"
        assert model.owned_by == "openai"

    def test_model_from_dict_with_missing_fields(self):
        """Test creating model from dict with missing fields."""
        data = {"id": "gpt-4"}
        model = Model.from_dict(data)
        assert model.id == "gpt-4"
        assert model.object == "model"
        assert model.created == 0
        assert model.owned_by == "unknown"


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

    def test_completion_request_to_dict_optional_fields(self):
        """Test converting completion request without optional fields."""
        request = CompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = request.to_dict()
        assert result == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }


class TestGatewayConfig:
    """Tests for the GatewayConfig dataclass."""

    def test_gateway_config_creation(self):
        """Test creating gateway configuration."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key="test-key",
            timeout=60,
        )
        assert config.downstream_url == "https://api.example.com"
        assert config.model_list_url == "https://models.dev/api/v1/models"
        assert config.default_model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.timeout == 60

    def test_gateway_config_to_dict(self):
        """Test converting config to dictionary."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key="test-key",
            timeout=60,
        )
        result = config.to_dict()
        assert result["downstream_url"] == "https://api.example.com"
        assert result["model_list_url"] == "https://models.dev/api/v1/models"
        assert result["default_model"] == "gpt-4"
        assert result["timeout"] == 60
        assert result["has_api_key"] is True

    def test_gateway_config_to_dict_no_api_key(self):
        """Test config to dict when no API key is set."""
        config = GatewayConfig(
            downstream_url="https://api.example.com",
            model_list_url="https://models.dev/api/v1/models",
            default_model="gpt-4",
            api_key=None,
            timeout=60,
        )
        result = config.to_dict()
        assert result["has_api_key"] is False

    def test_gateway_config_from_env_missing_downstream_url(self):
        """Test that missing DOWNSTREAM_URL raises error."""

        with pytest.raises(ValueError, match="DOWNSTREAM_URL"):
            with patch.dict(os.environ, {}, clear=True):
                GatewayConfig.from_env()

    def test_gateway_config_from_env_missing_default_model(self):
        """Test that missing DEFAULT_MODEL raises error."""

        with pytest.raises(ValueError, match="DEFAULT_MODEL"):
            with patch.dict(
                os.environ, {"DOWNSTREAM_URL": "https://api.example.com"}, clear=True
            ):
                GatewayConfig.from_env()

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
            assert config.downstream_url == "https://api.example.com"
            assert config.default_model == "gpt-4"

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
            },
            clear=True,
        ):
            config = GatewayConfig.from_env()
            assert config.downstream_url == "https://api.example.com"
            assert config.model_list_url == "https://custom.models.dev/api"
            assert config.api_key == "secret-key"
            assert config.timeout == 30
