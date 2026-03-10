"""
Agent module for the LangChain DataFrame Agent.

This module builds the LangChain agent that can answer questions about a
pandas DataFrame.  It wires together three LangChain components:

1. **ChatOllama** – the LLM backend.
   Original equivalent: custom HTTP calls to the Ollama REST API.

2. **create_pandas_dataframe_agent** – creates an AgentExecutor with a
   PythonREPLTool pre-configured for the supplied DataFrame.
   Original equivalent: manual tool definitions + agent reasoning loop.

3. **ConversationBufferMemory** – keeps track of the conversation history so
   follow-up questions work naturally.
   Original equivalent: manual message list management.
"""

import logging
from typing import Any

import pandas as pd
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import AgentType
from langchain_community.chat_models import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

from src.config import (
    AGENT_ALLOW_DANGEROUS_CODE,
    AGENT_MAX_ITERATIONS,
    AGENT_VERBOSE,
    OLLAMA_BASE_URL,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)

logger = logging.getLogger(__name__)


def build_llm() -> ChatOllama:
    """Instantiate the ChatOllama language model.

    ChatOllama is LangChain's wrapper around the Ollama local inference
    server.  In the original implementation the LLM was called via raw
    ``requests.post`` to the Ollama API; ChatOllama abstracts that away and
    integrates seamlessly with the rest of the LangChain ecosystem.

    Returns:
        A configured ``ChatOllama`` instance.
    """
    logger.info(
        "Initialising ChatOllama (model=%s, base_url=%s, temperature=%s).",
        OLLAMA_MODEL,
        OLLAMA_BASE_URL,
        OLLAMA_TEMPERATURE,
    )
    # ChatOllama replaces our custom HTTP-based LLM wrapper.
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_MAX_TOKENS,
    )


def build_agent(df: pd.DataFrame) -> Any:
    """Create a LangChain pandas dataframe agent for the given DataFrame.

    ``create_pandas_dataframe_agent`` is a convenience factory from
    ``langchain_experimental`` that:
      - Creates a Python REPL tool pre-loaded with ``df`` as a local variable.
      - Wraps the tool in an AgentExecutor with the supplied LLM.
      - Handles the full ReAct (Reason + Act) loop automatically.

    The original implementation manually built a reasoning loop, called the
    LLM, parsed the response, executed Python, and iterated.
    LangChain's AgentExecutor replaces that entire custom loop.

    Args:
        df: The pandas DataFrame the agent will analyse.

    Returns:
        A LangChain ``AgentExecutor`` ready to accept natural-language queries.
    """
    llm = build_llm()

    # ConversationBufferMemory stores the full chat history so the agent can
    # answer follow-up questions that reference earlier turns.
    # Original equivalent: a manually maintained list of {"role": …, "content": …} dicts.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    logger.info(
        "Building pandas dataframe agent (max_iterations=%d, verbose=%s).",
        AGENT_MAX_ITERATIONS,
        AGENT_VERBOSE,
    )

    # ZERO_SHOT_REACT_DESCRIPTION is chosen because:
    #   - It works with any LLM (including local Ollama models) via ReAct prompting.
    #   - It does not require function-calling support (unlike OPENAI_FUNCTIONS).
    #   - It is simpler and more compatible than STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    #     for straightforward pandas Q&A tasks.
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=AGENT_VERBOSE, # shows the full reasoning trace in the console, similar to our original print() calls.
        max_iterations=AGENT_MAX_ITERATIONS,
        allow_dangerous_code=AGENT_ALLOW_DANGEROUS_CODE,
        # Pass the memory so multi-turn conversations work.
        # The agent will prepend chat history to each prompt automatically.
        agent_executor_kwargs={"memory": memory, "handle_parsing_errors": True},
        
    )

    logger.info("Agent ready.")
    return agent


def ask(agent: Any, question: str) -> str:
    """Send a natural-language question to the agent and return its answer.

    Args:
        agent: The AgentExecutor returned by :func:`build_agent`.
        question: A plain-English question about the DataFrame.

    Returns:
        The agent's textual answer as a string.

    Raises:
        RuntimeError: When the agent raises an unexpected exception.
    """
    logger.debug("Sending question to agent: %s", question)
    try:
        # AgentExecutor.invoke() replaces our original manual call of
        # llm() + tool execution + result parsing.
        result = agent.invoke({"input": question})
        answer = result.get("output", str(result))
        logger.debug("Agent answer: %s", answer)
        return answer
    except Exception as exc:
        logger.error("Agent encountered an error: %s", exc)
        raise RuntimeError(f"Agent error: {exc}") from exc
