"""
Configuration module for the LangChain DataFrame Agent.

This module centralises all configuration settings, reading values from
environment variables with sensible defaults.  In the original custom
implementation these values were scattered across multiple files; here they
live in one place, making it easy to tune behaviour without touching code.
"""

import os


# ---------------------------------------------------------------------------
# Ollama / LLM settings
# ---------------------------------------------------------------------------

# The Ollama server endpoint.  Override with OLLAMA_BASE_URL when the server
# runs on a different host or port (e.g. inside Docker).
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The model served by Ollama.  Mistral is a capable open-source model that
# handles pandas-related reasoning well.  Change with OLLAMA_MODEL env var.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

# LLM temperature – 0 gives deterministic, reproducible answers which is
# preferable for data-analysis tasks.
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# Maximum tokens the model may generate in a single response.
OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))


# ---------------------------------------------------------------------------
# Agent settings
# ---------------------------------------------------------------------------

# Maximum number of reasoning steps the AgentExecutor is allowed to take
# before giving up.  Replaces the manual iteration cap in the original loop.
AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

# When True the agent prints each intermediate reasoning step to stdout.
# LangChain exposes this via verbose=True on the AgentExecutor; the original
# implementation achieved similar output with manual print() calls.
AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

# Allow potentially dangerous pandas operations (e.g. eval/exec).
# LangChain's create_pandas_dataframe_agent requires this flag when the
# agent needs to execute arbitrary Python to answer questions.
AGENT_ALLOW_DANGEROUS_CODE: bool = (
    os.getenv("AGENT_ALLOW_DANGEROUS_CODE", "true").lower() == "true"
)


# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------

# Path to the weather CSV file relative to the project root.
DATA_CSV_PATH: str = os.getenv("DATA_CSV_PATH", "data-set/weather-data.csv")


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------

# Python logging level for the application logger.
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Enable LangChain callback tracing logs for chain/tool/LLM lifecycle events.
CALLBACK_TRACE_ENABLED: bool = (
    os.getenv("CALLBACK_TRACE_ENABLED", "true").lower() == "true"
)

# Maximum characters logged for callback payload previews.
CALLBACK_TRACE_MAX_CHARS: int = int(os.getenv("CALLBACK_TRACE_MAX_CHARS", "600"))
