"""CLI entry point for the MCP LLM Gateway."""

from mcp_llm_gateway import mcp


def main() -> int:
    """Run the MCP server."""
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
