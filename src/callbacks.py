"""Callback handlers for LangChain execution tracing.

The handler in this module logs key chain/tool/LLM lifecycle events in a
compact, readable format so each run can be inspected from the session log.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


logger = logging.getLogger(__name__)


class AgentTraceCallbackHandler(BaseCallbackHandler):
    """Log LangChain callback events for debugging and observability."""

    def __init__(self, *, max_chars: int = 600) -> None:
        self._max_chars = max_chars

    def _clip(self, value: Any) -> str:
        text = str(value)
        if len(text) <= self._max_chars:
            return text
        return f"{text[: self._max_chars]}... [truncated {len(text) - self._max_chars} chars]"

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "unknown_chain"
        logger.info("[callback] chain.start name=%s inputs=%s", name, self._clip(inputs))

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        logger.info("[callback] chain.end outputs=%s", self._clip(outputs))

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error("[callback] chain.error error=%s", error)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        model_name = serialized.get("name") or serialized.get("id") or "unknown_llm"
        prompt_preview = prompts[0] if prompts else ""
        logger.info("[callback] llm.start model=%s prompt=%s", model_name, self._clip(prompt_preview))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generations = getattr(response, "generations", [])
        first_text = ""
        if generations and generations[0]:
            first_generation = generations[0][0]
            first_text = getattr(first_generation, "text", str(first_generation))
        logger.info("[callback] llm.end output=%s", self._clip(first_text))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error("[callback] llm.error error=%s", error)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name") or "unknown_tool"
        logger.info("[callback] tool.start name=%s input=%s", tool_name, self._clip(input_str))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        logger.info("[callback] tool.end output=%s", self._clip(output))

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error("[callback] tool.error error=%s", error)
