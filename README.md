# MCP LLM Gateway

> MCP-compatible LLM gateway that proxies completion requests to downstream OpenAI-compatible providers.

[![PyPI](https://img.shields.io/pypi/v/mcp-llm-gateway.svg)](https://pypi.org/project/mcp-llm-gateway/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-llm-gateway.svg)](https://pypi.org/project/mcp-llm-gateway/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

mcp-name: io.github.daedalus/mcp-llm-gateway

## Install

```bash
pip install mcp-llm-gateway
```

## Usage

### Configuration

Set the following environment variables:

- `DOWNSTREAM_URL`: Base URL for the OpenAI-compatible downstream API (required)
- `DEFAULT_MODEL`: Default model to use for completions (required)
- `MODEL_LIST_URL`: URL to fetch available models from (optional, defaults to models.dev)
- `API_KEY`: Optional API key for downstream (passthrough)
- `TIMEOUT`: Request timeout in seconds (optional, default: 60)

### MCP Server

Run the MCP server with stdio transport:

```bash
mcp-llm-gateway
```

### MCP Tools

The server exposes the following tools:

- `list_models()`: List all available models from the remote endpoint
- `complete(prompt, model, max_tokens, temperature)`: Send a completion request to the downstream LLM provider

### MCP Resources

- `models://list`: Returns the list of available models
- `config://info`: Returns current gateway configuration

## Development

```bash
git clone https://github.com/daedalus/mcp-llm-gateway.git
cd mcp-llm-gateway
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```

## API

### core.models

- `Model`: Dataclass representing an available LLM model
- `CompletionRequest`: Dataclass for completion request payloads
- `GatewayConfig`: Dataclass for gateway configuration

### adapters.http

- `HTTPAdapter`: HTTP client for downstream API communication
- `ModelListAdapter`: Adapter for fetching model list from remote endpoints

### services.gateway

- `ModelService`: Service for managing model discovery and caching
- `CompletionService`: Service for handling completion requests
- `ConfigService`: Service for managing gateway configuration