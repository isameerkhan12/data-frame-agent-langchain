"""
LangGraph-based Reflexion agent implementation.

This module implements a graph-based agent using LangGraph that implements
the full Reflexion pattern:

    start → execute → evaluate → (retry?) → reflect → memory ↻ execute
                            ↓
                          success ✓

KEY DIFFERENCES from ReAct-only implementation:
────────────────────────────────────────────────────

ReAct (Previous):
  Thought → Action → Observation → Thought → ...
  └─ Single linear chain
  └─ Errors become part of scratchpad
  └─ No explicit reflection step
  └─ Each query starts fresh (no learning)

Reflexion (New):
  execute → evaluate → reflect → [memory] → execute
  └─ Cyclic graph with explicit nodes
  └─ Dedicated reflection node analyzes WHY it failed
  └─ Stores lessons in persistent memory
  └─ Future attempts use past lessons as context
  └─ Configurable retry logic

This implementation uses LangGraph's StateGraph to manage the workflow,
making it explicit, debuggable, and easily extensible.
"""

import logging
from typing import Any, Sequence
from dataclasses import dataclass

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from src.agent import build_llm
from src.reflexion_memory import ReflexionMemory
from src.reflexion_evaluator import ReflexionEvaluator, ErrorInfo
from src.config import (
    AGENT_MAX_ITERATIONS,
    AGENT_VERBOSE,
    REFLEXION_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────────────────────

class ReflexionState(TypedDict):
    """
    The complete state passed through the LangGraph workflow.
    
    This TypedDict defines all data that flows through the graph nodes.
    Each node receives this state and may update it before passing to
    the next node.
    
    Attributes:
        input: The original user question
        messages: Conversation history with agent thoughts/actions
        dataframe: Reference to the DataFrame being analyzed
        attempt_number: Current retry attempt (1, 2, 3...)
        execution_result: Observation from tool execution
        error_info: Categorized error (if failed)
        reflections: List of reflections from previous attempts
        memory: The reflexion memory system
        final_answer: The answer to return to user
        last_node: Name of the last node executed (for debugging)
    """
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    dataframe: Any                              # pandas.DataFrame
    attempt_number: int
    execution_result: str
    error_info: Any                             # Optional[ErrorInfo]
    reflections: list[str]
    memory: Any                                 # ReflexionMemory
    final_answer: str
    last_node: str


def create_reflexion_graph(df: pd.DataFrame) -> tuple[Any, ReflexionMemory]:
    """
    Create the LangGraph reflexion agent.
    
    This builds the state machine with nodes for:
    1. execute – Run the agent and get observation
    2. evaluate – Check if succeeded or failed
    3. reflect – Analyze failure and generate lesson
    4. Route to END if successful or back to execute if retry
    
    Args:
        df: The pandas DataFrame to analyze
    
    Returns:
        Tuple of (compiled_graph, memory_instance)
        - compiled_graph: A runnable LangGraph agent
        - memory_instance: The ReflexionMemory instance for cross-query learning
    
    Configuration (from src/config.py):
        REFLEXION_MAX_ATTEMPTS: Max retry attempts (default: 5)
        AGENT_VERBOSE: Print detailed execution logs
    """
    
    # Initialize components
    llm = build_llm()
    memory = ReflexionMemory(max_lessons=50)
    python_repl = PythonREPLTool()
    
    logger.info(
        "Creating ReflexionGraph with LLM=%s, max_attempts=%d",
        llm.model,
        REFLEXION_MAX_ATTEMPTS,
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Node 1: EXECUTE – Run the DataFrame agent attempt
    # ─────────────────────────────────────────────────────────────────────
    
    def node_execute(state: ReflexionState) -> dict:
        """
        Execute node: Run one attempt of the agent.
        
        This node:
        1. Takes the user question (and past reflections if retrying)
        2. Builds a prompt with context from memory
        3. Runs the agent via LLM + Python REPL tool
        4. Returns the observation
        
        Process:
          input → [add memory context] → agent_prompt → LLM → tool → result
        
        Args:
            state: Current state (updated by previous nodes)
        
        Returns:
            Updated state with execution_result
        """
        logger.info(
            f"[EXECUTE] Attempt {state['attempt_number']}/{REFLEXION_MAX_ATTEMPTS}"
        )
        
        # Build the question with context from memory and reflections
        question = state["input"]
        
        # Add past lessons if any
        lessons_context = state["memory"].get_context(question, top_k=3)
        if lessons_context:
            logger.debug(f"[EXECUTE] Adding memory context from past lessons")
            question = question + lessons_context
        
        # Add reflections from previous attempts
        if state["reflections"]:
            reflections_text = "\n".join(state["reflections"])
            question = (
                question + 
                f"\n\n📌 PREVIOUS ATTEMPT FEEDBACK:\n{reflections_text}"
            )
            logger.debug(f"[EXECUTE] Added {len(state['reflections'])} reflection(s)")
        
        if AGENT_VERBOSE:
            logger.info(f"[EXECUTE] Question: {question[:300]}...")
        
        # Set DataFrame in REPL environment
        python_repl.locals["df"] = state["dataframe"]
        
        # Create prompt for LLM (asking it to generate Python code)
        prompt_text = (
            f"You are a pandas analysis expert. Answer this question by writing Python code.\n\n"
            f"Question: {question}\n\n"
            f"Use the 'df' variable (a pandas DataFrame already loaded).\n"
            f"Write only executable Python code. Provide the answer clearly."
        )
        
        try:
            # Get LLM response (should contain Python code)
            response = llm.invoke([HumanMessage(content=prompt_text)])
            code = response.content
            
            logger.debug(f"[EXECUTE] Generated code: {code[:200]}...")
            
            # Execute the code in the REPL
            result = python_repl.invoke(code)
            
            if AGENT_VERBOSE:
                logger.info(f"[EXECUTE] Result: {str(result)[:200]}")
            
            state["execution_result"] = str(result)
            state["last_node"] = "execute_success"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[EXECUTE] Exception: {error_msg[:200]}")
            state["execution_result"] = error_msg
            state["last_node"] = "execute_error"
        
        return state
    
    # ─────────────────────────────────────────────────────────────────────
    # Node 2: EVALUATE – Determine if execution succeeded or failed
    # ─────────────────────────────────────────────────────────────────────
    
    def node_evaluate(state: ReflexionState) -> dict:
        """
        Evaluate node: Determine if attempt succeeded or failed.
        
        This is the "critic" node that asks:
        "Did the agent succeed? Or did it encounter an error?"
        
        If failed, categorize the error for the reflection node.
        
        Process:
          execution_result → [check for errors] → error_info | success
        
        Args:
            state: Current state with execution_result
        
        Returns:
            Updated state with error_info (if failed)
        """
        logger.info("[EVALUATE] Analyzing execution result")
        
        result = state["execution_result"]
        
        # Check if the result contains an error
        if ReflexionEvaluator.is_failed(result):
            logger.warning(f"[EVALUATE] Failure detected: {result[:100]}")
            
            # Categorize the error
            error_info = ReflexionEvaluator.categorize_error(result)
            state["error_info"] = error_info
            state["last_node"] = "evaluate_failed"
            
        else:
            logger.info("[EVALUATE] Success! Setting final answer")
            state["final_answer"] = result
            state["last_node"] = "evaluate_success"
        
        return state
    
    # ─────────────────────────────────────────────────────────────────────
    # Node 3: REFLECT – Analyze failure and generate reflection
    # ─────────────────────────────────────────────────────────────────────
    
    def node_reflect(state: ReflexionState) -> dict:
        """
        Reflect node: Analyze the failure and generate a reflection.
        
        This node runs only if evaluation detected a failure.
        
        It:
        1. Creates a detailed analysis of why it failed
        2. Generates an actionable reflection
        3. Stores the lesson in persistent memory
        4. Updates state for potential retry
        
        This is the core of "learning" in Reflexion.
        
        Process:
          error_info + attempt_history
            → [generate reflection]
            → [store lesson]
            → reflection_text
        
        Args:
            state: Current state with error_info
        
        Returns:
            Updated state with reflections
        """
        logger.info("[REFLECT] 🧠 Generating reflection on failure")
        
        if not state.get("error_info"):
            logger.warning("[REFLECT] No error_info to reflect on")
            return state
        
        error_info = state["error_info"]
        
        # Generate reflection
        reflection = ReflexionEvaluator.generate_reflection(
            error_info=error_info,
            attempt_number=state["attempt_number"],
            previous_reflections=state["reflections"],
        )
        
        # Store lesson in memory for future queries
        state["memory"].add_lesson(
            problem=error_info.category,
            error_message=error_info.error_message,
            error_type=error_info.error_type,
            solution=reflection,
        )
        
        # Add to reflections list
        state["reflections"].append(reflection)
        
        logger.info(f"[REFLECT] 💾 Lesson stored. Reflection:\n{reflection[:200]}")
        state["last_node"] = "reflect"
        
        return state
    
    # ─────────────────────────────────────────────────────────────────────
    # Router: Decide whether to retry or finish
    # ─────────────────────────────────────────────────────────────────────
    
    def router_retry_or_end(state: ReflexionState) -> str:
        """
        Routing logic: Determine if we should retry or finish.
        
        Decision tree:
        - If execution succeeded (final_answer set) → go to END
        - If no error_info → go to END (success)
        - If error is unrecoverable → go to END
        - If max attempts reached → go to END
        - Otherwise → go back to EXECUTE (retry with reflection)
        
        This prevents infinite loops while allowing retries for
        recoverable errors.
        
        Args:
            state: Current state after evaluate/reflect
        
        Returns:
            Router decision: "execute" (retry) or END
        """
        # Success case
        if state.get("final_answer"):
            logger.info("[ROUTER] ✅ Success! Going to END")
            return END
        
        if not state.get("error_info"):
            logger.info("[ROUTER] No error detected. Going to END")
            return END
        
        error_info = state["error_info"]
        
        # Decide if should retry
        should_retry = ReflexionEvaluator.should_retry(
            error_info=error_info,
            attempt_number=state["attempt_number"],
            max_attempts=REFLEXION_MAX_ATTEMPTS,
        )
        
        if should_retry:
            logger.info(
                f"[ROUTER] 🔄 Error is recoverable. Retrying "
                f"(attempt {state['attempt_number'] + 1}/{REFLEXION_MAX_ATTEMPTS})"
            )
            state["attempt_number"] += 1
            state["error_info"] = None  # Reset for next attempt
            return "execute"
        else:
            logger.warning("[ROUTER] ❌ Error is fatal or max attempts reached. Ending.")
            state["final_answer"] = (
                f"Failed after {state['attempt_number']} attempts. "
                f"Error: {error_info.error_type}. {error_info.error_message[:100]}"
            )
            return END
    
    # ─────────────────────────────────────────────────────────────────────
    # Build the Graph
    # ─────────────────────────────────────────────────────────────────────
    
    graph = StateGraph(ReflexionState)
    
    # Add nodes
    graph.add_node("execute", node_execute)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("reflect", node_reflect)
    
    # Define edges
    graph.add_edge(START, "execute")           # Start with execution
    graph.add_edge("execute", "evaluate")      # Then evaluate result
    
    # Conditional: success or failure in evaluate
    graph.add_conditional_edges(
        "evaluate",
        lambda state: "reflect" if state.get("error_info") else END,
        {
            "reflect": "reflect",
            END: END,
        },
    )
    
    graph.add_conditional_edges(
        "reflect",
        router_retry_or_end,
        {
            "execute": "execute",
            END: END,
        },
    )
    
    logger.info("[GRAPH] LangGraph reflexion state machine created")
    
    # Compile the graph
    agent = graph.compile()
    
    return agent, memory


def ask_with_reflexion(
    agent: Any,
    memory: ReflexionMemory,
    question: str,
    dataframe: pd.DataFrame,
) -> str:
    """
    Ask the reflexion agent a question.
    
    Initializes the state and invokes the graph. The graph automatically
    handles the full reflexion loop (execute → evaluate → reflect → retry).
    
    Args:
        agent: The compiled LangGraph agent
        memory: The shared reflexion memory
        question: User question
        dataframe: The pandas DataFrame to analyze
    
    Returns:
        The final answer after potentially multiple attempts
    """
    logger.info("[GRAPH] Starting reflexion agent invocation")
    
    initial_state = {
        "input": question,
        "messages": [],
        "dataframe": dataframe,
        "attempt_number": 1,
        "execution_result": "",
        "error_info": None,
        "reflections": [],
        "memory": memory,
        "final_answer": "",
        "last_node": "start",
    }
    
    result = agent.invoke(initial_state)
    
    final_answer = result.get("final_answer", "No answer generated")
    logger.info(f"[GRAPH] Reflexion completed. Last node: {result.get('last_node')}")
    
    return final_answer
