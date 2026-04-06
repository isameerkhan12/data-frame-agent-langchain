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
    CALLBACK_TRACE_ENABLED,
    CALLBACK_TRACE_MAX_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    REFLEXION_ENABLED,
    REFLEXION_MAX_ATTEMPTS,
    REFLEXION_MAX_LESSONS,
)
from src.callbacks import AgentTraceCallbackHandler

logger = logging.getLogger(__name__)


REACT_PREFIX = """You are a pandas dataframe analysis agent.
You MUST follow this exact loop format and nothing else:

Thought: <short reasoning>
Action: python_repl_ast
Action Input: <single valid Python expression using df>
Observation: <tool result>

Repeat Thought/Action/Action Input/Observation as needed.

When finished, output ONLY:
Final Answer: <concise answer with reasoning from observations>

Rules:
- Only allowed tool name is exactly python_repl_ast
- Never add apologies or extra text before Action
- Never wrap Action Input in markdown fences
- Keep Action Input executable Python only
- If a previous parse failed, continue with the exact format above
"""

REACT_SUFFIX = """Begin.

Question: {input}
{agent_scratchpad}
"""

PARSING_ERROR_HINT = (
    "Your previous message did not follow the required format. "
    "Return ONLY Thought, Action, Action Input, Observation, or Final Answer. "
    "Tool name must be exactly python_repl_ast."
)


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

    callbacks = []
    if CALLBACK_TRACE_ENABLED:
        callbacks.append(AgentTraceCallbackHandler(max_chars=CALLBACK_TRACE_MAX_CHARS))
        logger.info(
            "Callback tracing enabled (max_chars=%d).",
            CALLBACK_TRACE_MAX_CHARS,
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
        allow_dangerous_code=AGENT_ALLOW_DANGEROUS_CODE, #True → LangChain allows eval/exec (needed for pandas analysis)
        prefix=REACT_PREFIX,
        suffix=REACT_SUFFIX,
        include_df_in_prompt=None,
        # Pass the memory so multi-turn conversations work.
        # The agent will prepend chat history to each prompt automatically.
        agent_executor_kwargs={
            "memory": memory,
            "handle_parsing_errors": PARSING_ERROR_HINT, # ← EXPLICIT error handling (looping back if LLM format mistakes occur)
            "callbacks": callbacks, # ← Tool errors logged here
        },
        # prefix=
        
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


def ask_reflexion(agent: Any, memory: Any, question: str, df: pd.DataFrame) -> str:
    """
    Ask using the Reflexion graph (with learning & persistence).
    
    This is the new mode for complex questions where the agent should
    learn from failures and retry with reflection. It explicitly implements
    the Reflexion pattern with reflection nodes and persistent memory.
    
    Configuration:
        REFLEXION_ENABLED: Must be True to use this mode
        REFLEXION_MAX_ATTEMPTS: Number of retry attempts (default: 5)
        REFLEXION_MAX_LESSONS: Max stored lessons (default: 50)
    
    How it works:
        1. EXECUTE: Run the agent attempt
        2. EVALUATE: Check if success or error
        3. If error → REFLECT: Analyze and store lesson
        4. ROUTER: Retry if learnable error, else give up
        5. Lessons persist to next query
    
    Args:
        agent: The compiled LangGraph reflexion agent
        memory: The shared ReflexionMemory instance for cross-query learning
        question: User question
        df: The pandas DataFrame to analyze
    
    Returns:
        The final answer after potentially multiple attempts
    
    Raises:
        RuntimeError: If the agent encounters a fatal error
    """
    logger.debug("Using Reflexion mode for question: %s", question)
    try:
        from src.reflexion_graph import ask_with_reflexion
        result = ask_with_reflexion(agent, memory, question, df)
        logger.debug("Reflexion answer: %s", result)
        return result
    except Exception as exc:
        logger.error("Reflexion agent error: %s", exc)
        raise RuntimeError(f"Reflexion agent error: {exc}") from exc
