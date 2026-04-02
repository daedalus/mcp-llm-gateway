# SPEC.md — MCP LLM Gateway

## Purpose

An MCP-compatible server that acts as an LLM gateway, proxying completion requests to downstream OpenAI-compatible providers and dynamically exposing available models from remote model discovery endpoints (e.g., models.dev).

## Scope

### In Scope
- MCP server implementing the Model Context Protocol (stdio transport)
- Dynamic model discovery from remote endpoints (e.g., models.dev)
- Proxying chat/completion requests to downstream OpenAI-compatible APIs
- Exposing tools/resources for model listing and completion
- Configuration via config.yaml file
- Support for multiple providers with their respective data

### Not In Scope
- Authentication/authorization (passthrough to downstream)
- Caching or rate limiting
- Multiple transport types (stdio only)
- Custom model registration UI

## Configuration

### config.yaml

The gateway is configured via a `config.yaml` file with the following structure:

```yaml
# Optional: Model list endpoint for discovering available models
model_list_url: "https://models.dev/api/v1/models"

# Optional: Cache TTL for model list in seconds (default: 300)
cache_ttl: 300

# Required: List of providers
providers:
  - id: "openai"               # Unique provider identifier
    name: "OpenAI"             # Human-readable name
    type: "openai"             # Provider type (openai/anthropic/etc)
    base_url: "https://api.openai.com/v1"  # API base URL
    api_key: "${OPENAI_API_KEY}"  # API key (supports env var interpolation)
    default_model: "gpt-4"    # Default model for this provider
    timeout: 60                # Request timeout in seconds (optional)
    enabled: true              # Enable/disable provider (optional, default: true)

  - id: "anthropic"
    name: "Anthropic"
    type: "anthropic"
    base_url: "https://api.anthropic.com/v1"
    api_key: "${ANTHROPIC_API_KEY}"
    default_model: "claude-3-sonnet-20240229"
    enabled: true
```

### Environment Variable Interpolation

The config.yaml supports environment variable interpolation using `${VAR_NAME}` syntax. This allows sensitive data like API keys to be loaded from environment variables.

### Configuration Loading Priority

1. `config.yaml` in current working directory
2. `config.yaml` at `/etc/mcp-llm-gateway/config.yaml`
3. Environment variables (legacy fallback): `DOWNSTREAM_URL`, `DEFAULT_MODEL`, `MODEL_LIST_URL`, `API_KEY`

## Public API / Interface

### MCP Tools

`list_models(provider: str | null = null)`
- Lists all available models from the configured providers
- Args:
  - `provider`: Optional provider ID to filter models
- Returns: List of model objects with id, name, and metadata

`complete(prompt: str, model: str | null = null, provider: str | null = null, max_tokens: int | null = null, temperature: float | null = null)`
- Proxies a completion request to the downstream provider
- Args:
  - `prompt`: The input prompt
  - `model`: Model ID (optional, uses provider default if not specified)
  - `provider`: Provider ID (optional, uses first enabled provider if not specified)
  - `max_tokens`: Maximum tokens to generate
  - `temperature`: Sampling temperature
- Returns: Completion response from downstream provider

### MCP Resources

`models://list`
- Resource URI that returns the list of available models
- Returns: JSON array of model objects

`config://info`
- Resource URI that returns current gateway configuration
- Returns: Configuration details (providers, etc.)

`providers://list`
- Resource URI that returns the list of configured providers
- Returns: JSON array of provider objects

## Data Formats

### Provider Object
```yaml
id: string
name: string
type: string
base_url: string
default_model: string
timeout: number
enabled: boolean
```

### Model Object
```json
{
  "id": "string",
  "object": "model",
  "created": 1234567890,
  "owned_by": "string",
  "provider_id": "string"
}
```

### Completion Request (OpenAI format)
```json
{
  "model": "string",
  "messages": [{"role": "user", "content": "string"}],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

### Completion Response (OpenAI format)
```json
{
  "id": "string",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "string",
  "choices": [...],
  "usage": {...}
}
```

## Edge Cases

1. **Downstream API unavailable**: Return meaningful error to client
2. **Invalid model requested**: Return 404-style error with available models list
3. **Empty model list from remote**: Fall back to configured default model
4. **Network timeout**: Propagate timeout error to client
5. **Malformed response from downstream**: Return error response, don't crash
6. **Provider disabled**: Return error indicating provider is not available
7. **Unknown provider**: Return error with list of valid provider IDs
8. **Missing config.yaml**: Fall back to environment variable configuration

## Performance & Constraints

- Timeout for downstream requests: configurable per provider (default 60 seconds)
- No authentication validation (passthrough)
- Minimal caching (model list cached for 5 minutes by default)
- Python 3.11+ required