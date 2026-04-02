"""Test fixtures and configuration for pytest."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Create a mock gateway configuration."""
    from mcp_llm_gateway.core.models import GatewayConfig

    return GatewayConfig(
        downstream_url="https://api.example.com",
        model_list_url="https://models.dev/api/v1/models",
        default_model="gpt-4",
        api_key="test-key",
        timeout=60,
    )


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1234567890,
                "owned_by": "openai",
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1234567890,
                "owned_by": "openai",
            },
        ]
    }
    return response


@pytest.fixture
def mock_model_list_response():
    """Create a mock model list response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [
        {"id": "gpt-4", "object": "model", "created": 1234567890, "owned_by": "openai"},
        {
            "id": "claude-3",
            "object": "model",
            "created": 1234567890,
            "owned_by": "anthropic",
        },
    ]
    return response


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
