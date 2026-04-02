"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mcp_llm_gateway.core.config import load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "model_list_url": "https://models.dev/api/v1/models",
            "cache_ttl": 300,
            "providers": [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert len(config.providers) == 1
            assert config.providers[0].id == "openai"
            assert config.model_list_url == "https://models.dev/api/v1/models"
            assert config.cache_ttl == 300
        finally:
            Path(config_path).unlink()

    def test_load_config_fallback_to_env(self):
        """Test fallback to environment variables when no config file."""
        with patch.dict(
            os.environ,
            {
                "DOWNSTREAM_URL": "https://api.example.com",
                "DEFAULT_MODEL": "gpt-4",
            },
            clear=True,
        ):
            config = load_config()
            assert len(config.providers) == 1
            assert config.providers[0].base_url == "https://api.example.com"

    def test_load_config_missing_provider_raises(self):
        """Test that missing providers raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"providers": []}, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="At least one provider"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_config_with_env_interpolation(self):
        """Test loading config with environment variable interpolation."""
        os.environ["TEST_API_KEY"] = "sk-test-key"

        config_data = {
            "providers": [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "${TEST_API_KEY}",
                    "default_model": "gpt-4",
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config.providers[0].api_key == "sk-test-key"
        finally:
            Path(config_path).unlink()
            del os.environ["TEST_API_KEY"]

    def test_load_config_invalid_yaml_raises(self):
        """Test that invalid YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid:\n  nested: value\n")
            config_path = f.name

        try:
            with pytest.raises((ValueError, yaml.YAMLError)):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
