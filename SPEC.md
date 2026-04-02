# SPEC.md — MCP LLM Gateway

## Purpose

An MCP-compatible server that acts as an LLM gateway, proxying completion requests to downstream OpenAI-compatible providers and dynamically exposing available models from remote model discovery endpoints (e.g., models.dev).

## Scope

### In Scope
- MCP server implementing the Model Context Protocol (stdio transport)
- Dynamic model discovery from remote endpoints (e.g., models.dev)
- Proxying chat/completion requests to downstream OpenAI-compatible APIs
- Exposing tools/resources for model listing and completion
- Configuration via environment variables

### Not In Scope
- Authentication/authorization (passthrough to downstream)
- Caching or rate limiting
- Multiple transport types (stdio only)
- Custom model registration UI

## Public API / Interface

### MCP Tools

`list_models()`
- Lists all available models from the remote endpoint
- Returns: List of model objects with id, name, and metadata

`complete(prompt: str, model: str | None = None, max_tokens: int | None = None, temperature: float | None = None)`
- Proxies a completion request to the downstream provider
- Args:
  - `prompt`: The input prompt
  - `model`: Model ID (optional, uses default if not specified)
  - `max_tokens`: Maximum tokens to generate
  - `temperature`: Sampling temperature
- Returns: Completion response from downstream provider

### MCP Resources

`models://list`
- Resource URI that returns the list of available models
- Returns: JSON array of model objects

`config://info`
- Resource URI that returns current gateway configuration
- Returns: Configuration details (endpoint URL, etc.)

### Environment Configuration

- `DOWNSTREAM_URL`: Base URL for the OpenAI-compatible downstream API (required)
- `MODEL_LIST_URL`: URL to fetch available models from (optional, defaults to models.dev)
- `DEFAULT_MODEL`: Default model to use for completions (required)
- `API_KEY`: Optional API key for downstream (passthrough)

## Data Formats

### Model Object
```json
{
  "id": "string",
  "object": "model",
  "created": 1234567890,
  "owned_by": "string"
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

## Performance & Constraints

- Timeout for downstream requests: 60 seconds
- No authentication validation (passthrough)
- Minimal caching (model list cached for 5 minutes)
- Python 3.11+ required