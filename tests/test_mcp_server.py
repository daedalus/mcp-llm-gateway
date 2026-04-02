"""Tests for MCP server."""

import os
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_gateway import mcp_server


class TestMCPServer:
    """Tests for MCP server functions."""

    @pytest.fixture(autouse=True)
    def reset_services(self):
        """Reset global service state before each test."""
        mcp_server._model_service = None
        mcp_server._completion_service = None
        mcp_server._config_service = None
        yield
        mcp_server._model_service = None
        mcp_server._completion_service = None
        mcp_server._config_service = None

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        mock_model_service = MagicMock()
        mock_completion_service = MagicMock()
        mock_config_service = MagicMock()

        mock_model_service.list_models.return_value = [
            MagicMock(to_dict=lambda: {"id": "gpt-4", "object": "model"})
        ]
        mock_completion_service.complete.return_value = {
            "id": "chatcmpl-123",
            "choices": [],
        }
        mock_config_service.get_config.return_value = {
            "downstream_url": "https://api.example.com",
            "default_model": "gpt-4",
        }

        return mock_model_service, mock_completion_service, mock_config_service

    @patch.dict(
        os.environ,
        {
            "DOWNSTREAM_URL": "https://api.example.com",
            "DEFAULT_MODEL": "gpt-4",
        },
        clear=True,
    )
    def test_list_models_tool(self, mock_services):
        """Test the list_models MCP tool."""
        mock_model, mock_completion, mock_config = mock_services

        with (
            patch.object(mcp_server, "ModelService", return_value=mock_model),
            patch.object(mcp_server, "CompletionService", return_value=mock_completion),
            patch.object(mcp_server, "ConfigService", return_value=mock_config),
        ):
            # Re-import to get fresh global state after patching
            result = mcp_server.list_models()
            assert len(result) == 1
            assert result[0]["id"] == "gpt-4"

    @patch.dict(
        os.environ,
        {
            "DOWNSTREAM_URL": "https://api.example.com",
            "DEFAULT_MODEL": "gpt-4",
        },
        clear=True,
    )
    def test_complete_tool(self, mock_services):
        """Test the complete MCP tool."""
        mock_model, mock_completion, mock_config = mock_services

        with (
            patch.object(mcp_server, "ModelService", return_value=mock_model),
            patch.object(mcp_server, "CompletionService", return_value=mock_completion),
            patch.object(mcp_server, "ConfigService", return_value=mock_config),
        ):
            result = mcp_server.complete("Hello, world!", model="gpt-4", max_tokens=100)
            assert "id" in result

    @patch.dict(
        os.environ,
        {
            "DOWNSTREAM_URL": "https://api.example.com",
            "DEFAULT_MODEL": "gpt-4",
        },
        clear=True,
    )
    def test_models_resource(self, mock_services):
        """Test the models list resource."""
        mock_model, mock_completion, mock_config = mock_services

        with (
            patch.object(mcp_server, "ModelService", return_value=mock_model),
            patch.object(mcp_server, "CompletionService", return_value=mock_completion),
            patch.object(mcp_server, "ConfigService", return_value=mock_config),
        ):
            result = mcp_server.models_list()
            assert len(result) == 1
            assert result[0]["id"] == "gpt-4"

    @patch.dict(
        os.environ,
        {
            "DOWNSTREAM_URL": "https://api.example.com",
            "DEFAULT_MODEL": "gpt-4",
        },
        clear=True,
    )
    def test_config_resource(self, mock_services):
        """Test the config info resource."""
        mock_model, mock_completion, mock_config = mock_services

        with (
            patch.object(mcp_server, "ModelService", return_value=mock_model),
            patch.object(mcp_server, "CompletionService", return_value=mock_completion),
            patch.object(mcp_server, "ConfigService", return_value=mock_config),
        ):
            result = mcp_server.config_info()
            assert result["downstream_url"] == "https://api.example.com"
            assert result["default_model"] == "gpt-4"

    def test_main_function(self):
        """Test main function runs mcp.run()."""
        with patch.object(mcp_server.mcp, "run"):
            result = mcp_server.main()
            assert result == 0
