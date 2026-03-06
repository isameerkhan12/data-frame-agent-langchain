"""
Entry point for the LangChain DataFrame Agent.

Usage
-----
Interactive mode (default):
    python main.py

Single-question mode:
    python main.py --question "Which city had the highest temperature?"

Options:
    --question TEXT   Ask a single question and exit.
    --log-level LEVEL Python logging level (DEBUG, INFO, WARNING, ERROR).
                      Can also be set via the LOG_LEVEL environment variable.
    --help            Show this help message and exit.

Environment variables (all optional):
    OLLAMA_BASE_URL          Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL             Model name       (default: mistral)
    OLLAMA_TEMPERATURE       LLM temperature  (default: 0)
    OLLAMA_MAX_TOKENS        Max output tokens (default: 2048)
    AGENT_MAX_ITERATIONS     AgentExecutor iteration cap (default: 10)
    AGENT_VERBOSE            Print chain-of-thought steps (default: false)
    AGENT_ALLOW_DANGEROUS_CODE  Allow Python eval in agent (default: true)
    DATA_CSV_PATH            Path to weather CSV (default: data-set/weather-data.csv)
    LOG_LEVEL                Logging level    (default: INFO)
"""

import argparse
import logging
import sys

from src.config import LOG_LEVEL
from src.app import run


def _configure_logging(level: str) -> None:
    """Set up root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangChain-based pandas DataFrame agent powered by Mistral via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--question",
        metavar="TEXT",
        default=None,
        help="Ask a single question and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    exit_code = run(question=args.question)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
