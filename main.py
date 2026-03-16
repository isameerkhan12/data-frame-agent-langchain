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
import atexit
from datetime import datetime
import logging
from pathlib import Path
import sys

from src.config import LOG_LEVEL
from src.app import run


class _TeeStream:
    """Write stream output to both terminal and a log file."""

    def __init__(self, primary_stream, log_stream) -> None:
        self._primary_stream = primary_stream
        self._log_stream = log_stream

    def write(self, text: str) -> int:
        self._primary_stream.write(text)
        self._log_stream.write(text)
        return len(text)

    def flush(self) -> None:
        self._primary_stream.flush()
        self._log_stream.flush()

    def isatty(self) -> bool:
        return self._primary_stream.isatty()


class _AnsiLevelFormatter(logging.Formatter):
    """Formatter that applies ANSI colors per log level."""

    _RESET = "\x1b[0m"
    _LEVEL_COLORS = {
        "DEBUG": "\x1b[36m",     # Cyan
        "INFO": "\x1b[32m",      # Green
        "WARNING": "\x1b[33m",   # Yellow
        "ERROR": "\x1b[31m",     # Red
        "CRITICAL": "\x1b[1;31m",  # Bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        base_line = super().format(record)
        color = self._LEVEL_COLORS.get(record.levelname, "")
        if not color:
            return base_line
        return f"{color}{base_line}{self._RESET}"


def _setup_results_capture() -> Path:
    """Create results folder and mirror stdout/stderr to a timestamped log file."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"run_{timestamp}.log"
    # newline="" preserves exact line endings and avoids Windows \r\r\n issues.
    log_file = log_path.open("a", encoding="utf-8", buffering=1, newline="")

    # Ensure console output remains visible while persisting the full session.
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    def _cleanup() -> None:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    return log_path


def _configure_logging(level: str) -> None:
    """Set up root logger with a consistent, ANSI-colored format."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level))

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(getattr(logging, level))
    stream_handler.setFormatter(
        _AnsiLevelFormatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(stream_handler)


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
    log_path = _setup_results_capture()
    _configure_logging(args.log_level)
    print(f"Session logs are being saved to: {log_path}")

    exit_code = run(question=args.question)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
