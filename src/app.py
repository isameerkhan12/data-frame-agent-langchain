"""
Application module for the LangChain DataFrame Agent.

Provides two interaction modes:

* **Single-question mode** – answer one question passed via ``--question``
  and exit.  Useful for scripting and automation.

* **Interactive (REPL) mode** – a read-evaluate-print loop that keeps the
  agent alive so follow-up questions can reference earlier answers via
  ConversationBufferMemory.

In the original implementation these modes were implemented with ad-hoc
``if/else`` blocks and a bare ``while True`` loop.  Here the logic is
cleanly separated into typed functions, each with proper error handling and
logging.
"""

import logging
import sys
from typing import Any

from src.agent import ask, build_agent
from src.data_loader import load_weather_data

logger = logging.getLogger(__name__)

# Prompt shown to the user at the start of an interactive session.
_WELCOME = (
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║        LangChain Weather DataFrame Agent  (Mistral)          ║\n"
    "║  Type a question about the weather data, or 'exit' to quit.  ║\n"
    "╚══════════════════════════════════════════════════════════════╝\n"
)


def _print_answer(answer: str) -> None:
    """Pretty-print the agent's answer to stdout."""
    print(f"\n🤖  {answer}\n")


def run_single_question(question: str) -> int:
    """Answer a single question and exit.

    Args:
        question: The natural-language question to send to the agent.

    Returns:
        An exit code: 0 on success, 1 on error.
    """
    logger.info("Running in single-question mode.")
    try:
        df = load_weather_data()
        agent = build_agent(df)
        answer = ask(agent, question)
        _print_answer(answer)
        return 0
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        logger.error("Agent error: %s", exc)
        print(f"Agent error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


def run_interactive(agent: Any) -> None:
    """Start an interactive REPL session.

    The agent is kept alive between questions so ConversationBufferMemory
    accumulates context, enabling follow-up questions like "What about in
    Chicago?" after asking "What was the hottest day?".

    Args:
        agent: The AgentExecutor returned by :func:`src.agent.build_agent`.
    """
    print(_WELCOME)
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q", "bye"}:
            print("Goodbye!")
            break

        try:
            answer = ask(agent, question)
            _print_answer(answer)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except Exception as exc:
            logger.exception("Unexpected error during interactive session: %s", exc)
            print(f"Unexpected error: {exc}", file=sys.stderr)


def run(question: str | None = None) -> int:
    """Main entry point for the application.

    Args:
        question: When provided, the agent answers this single question and
                  exits.  When ``None``, an interactive REPL is started.

    Returns:
        An exit code: 0 on success, 1 on error.
    """
    if question is not None:
        return run_single_question(question)

    # Interactive mode – load data and build agent once, then loop.
    logger.info("Running in interactive mode.")
    try:
        df = load_weather_data()
        agent = build_agent(df)
        run_interactive(agent)
        return 0
    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Failed to initialise agent: %s", exc)
        print(f"Failed to initialise agent: {exc}", file=sys.stderr)
        return 1
