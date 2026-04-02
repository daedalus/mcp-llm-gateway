"""Logging configuration for the MCP LLM Gateway."""

import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    log_file: str | None = None,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up logging configuration for the gateway.

    Args:
        log_file: Path to log file. If None, uses default location.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format string for log messages.

    Returns:
        Configured logger instance.
    """
    if log_file is None:
        log_dir = Path(os.environ.get("LOG_DIR", "/tmp/mcp-llm-gateway-logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "mcp-llm-gateway.log")

    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level_value,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("mcp-llm-gateway")
    logger.setLevel(log_level_value)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Logger instance.
    """
    if name is None:
        return logging.getLogger("mcp-llm-gateway")
    return logging.getLogger(f"mcp-llm-gateway.{name}")


class GatewayLogger:
    """Structured logger for the gateway with contextual information."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the gateway logger."""
        self._logger = get_logger(name)
        self._context: dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Set contextual information for logging."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear contextual information."""
        self._context.clear()

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self._context:
            context_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
            return f"{message} | {context_str}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Log debug message."""
        self._logger.debug(self._format_message(message), extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Log info message."""
        self._logger.info(self._format_message(message), extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Log warning message."""
        self._logger.warning(self._format_message(message), extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Log error message."""
        self._logger.error(self._format_message(message), extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Log critical message."""
        self._logger.critical(self._format_message(message), extra=kwargs)

    def log_request(  # noqa: ANN401
        self,
        provider: str,
        model: str,
        prompt: str | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log a completion request."""
        status = "success" if success else "failed"
        msg = f"Request | provider={provider} | model={model} | status={status}"
        if prompt:
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            msg += f" | prompt={prompt_preview}"
        if error:
            msg += f" | error={error}"
        if duration_ms is not None:
            msg += f" | duration={duration_ms:.2f}ms"

        if success:
            self.info(msg)
        else:
            self.error(msg)

    def log_model_fallback(
        self,
        provider: str,
        failed_model: str,
        fallback_model: str,
    ) -> None:
        """Log model fallback."""
        self.warning(
            f"Model fallback | provider={provider} | "
            f"failed={failed_model} | fallback={fallback_model}"
        )

    def log_config_loaded(self, providers: list[str], config_path: str | None) -> None:
        """Log configuration loaded."""
        self.info(
            f"Config loaded | path={config_path or 'env'} | "
            f"providers={', '.join(providers)}"
        )
