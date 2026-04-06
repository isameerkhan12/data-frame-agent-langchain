"""
Evaluator module for detecting failures and generating reflections.

This module implements the core reflection logic:
1. Failure detection – identify if an attempt failed
2. Error categorization – classify failure type
3. Reflection generation – analyze root cause and suggest improvements

These functions are called within the LangGraph nodes to drive the
reflexion loop.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorInfo:
    """Container for categorized error information."""
    
    def __init__(
        self,
        error_type: str,
        error_message: str,
        category: str,
        is_recoverable: bool,
    ):
        self.error_type = error_type              # Python error class
        self.error_message = error_message        # Full error text
        self.category = category                  # High-level category
        self.is_recoverable = is_recoverable      # Can agent recover?


class ReflexionEvaluator:
    """
    Evaluates agent responses and determines if they indicate failure.
    
    This is the "critic" that examines what the agent did and says whether
    it was successful or not. If failed, it categorizes the error and
    generates a reflection to guide the next attempt.
    
    Key responsibilities:
    - Detect if an Observation contains an error
    - Categorize error type (column error, type error, syntax, iteration limit)
    - Generate specific, actionable reflections
    - Determine if error is recoverable or fatal
    """
    
    # Patterns for common pandas/Python errors
    ERROR_PATTERNS = {
        "column_not_found": {
            "patterns": [r"KeyError.*(?:column|key)", r"KeyError:\s*'[^']*'"],
            "category": "column_not_found",
            "suggestion": "Always run df.columns.tolist() first to verify column names. Use lowercase, underscore-separated names without spaces or special characters.",
        },
        "type_error": {
            "patterns": [r"TypeError", r"cannot perform reduce"],
            "category": "type_error",
            "suggestion": "Check data types with df.dtypes before performing operations. Use .astype() to convert if needed.",
        },
        "syntax_error": {
            "patterns": [r"SyntaxError", r"invalid syntax"],
            "category": "syntax_error",
            "suggestion": "Review Python syntax. Ensure imports are correct, parentheses balanced, and string quotes matched.",
        },
        "attribute_error": {
            "patterns": [r"AttributeError"],
            "category": "attribute_error",
            "suggestion": "Verify the object/method exists. Check available methods with dir(obj) or help(obj).",
        },
        "iteration_limit": {
            "patterns": [r"iteration limit", r"stopped due to", r"max_iterations"],
            "category": "iteration_limit",
            "suggestion": "Question is too complex. Break it into smaller sub-questions or provide more specific dataframe filtering criteria.",
        },
        "index_error": {
            "patterns": [r"IndexError", r"index out of range"],
            "category": "index_error",
            "suggestion": "Check DataFrame dimensions with df.shape before accessing by index. Verify row/column exists.",
        },
    }
    
    @staticmethod
    def is_failed(observation: str) -> bool:
        """
        Determine if the Observation indicates failure.
        
        An observation is considered failed if it contains:
        - Python error messages (KeyError, TypeError, etc.)
        - Agent timeout messages ("iteration limit")
        - Explicit failure indicators
        
        Args:
            observation: The observation text from tool execution
        
        Returns:
            True if observation indicates failure, False if successful
        """
        failure_indicators = [
            "Error",
            "error",
            "exception",
            "failed",
            "stopped due to",
            "iteration limit",
            "Traceback",
        ]
        
        return any(indicator in observation for indicator in failure_indicators)
    
    @staticmethod
    def categorize_error(error_message: str) -> ErrorInfo:
        """
        Categorize the error and provide structured information.
        
        Matches error message against known patterns to determine:
        - Error type (KeyError, TypeError, etc.)
        - Category (high-level group: column_not_found, type_error, etc.)
        - Whether it's likely recoverable
        - Specific suggestion for fixing
        
        Args:
            error_message: The error message from the tool
        
        Returns:
            ErrorInfo object with categorized details
        """
        # Extract error type from message
        error_type = "Unknown"
        if "Error:" in error_message:
            error_type = error_message.split("Error:")[0].split()[-1]
        
        # Match against patterns
        matched_category = "generic_error"
        suggestion = "Review the error message and adjust your approach."
        
        for error_key, error_config in ReflexionEvaluator.ERROR_PATTERNS.items():
            for pattern in error_config["patterns"]:
                if re.search(pattern, error_message, re.IGNORECASE):
                    matched_category = error_config["category"]
                    suggestion = error_config["suggestion"]
                    break
        
        # Most errors are recoverable (except syntax errors)
        is_recoverable = "Syntax" not in error_type
        
        logger.info(
            f"Error categorized: {error_type} → {matched_category} "
            f"(recoverable={is_recoverable})"
        )
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            category=matched_category,
            is_recoverable=is_recoverable,
        )
    
    @staticmethod
    def generate_reflection(
        error_info: ErrorInfo,
        attempt_number: int,
        previous_reflections: list[str] = None,
    ) -> str:
        """
        Generate a reflection based on failure analysis.
        
        Creates a structured reflection that:
        1. Acknowledges what was tried
        2. Explains why it failed
        3. Suggests specific improvements
        4. References previous reflections if relevant
        
        Args:
            error_info: Categorized error information
            attempt_number: Which attempt this is (1, 2, 3...)
            previous_reflections: Reflections from prior attempts
        
        Returns:
            A reflection string that will be added to the agent's prompt
        """
        reflection = (
            f"REFLECTION ON ATTEMPT {attempt_number}:\n"
            f"- Error Type: {error_info.error_type}\n"
            f"- Category: {error_info.category}\n"
            f"- Problem: {error_info.error_message[:100]}\n"
            f"- Suggestion: {error_info.suggestion}\n"
        )
        
        # Reference previous attempts if we have patterns
        if previous_reflections and len(previous_reflections) > 1:
            reflection += (
                f"\n⚠️  This is attempt {attempt_number}. "
                f"You tried something similar before. "
                f"Please adapt your approach."
            )
        
        return reflection
    
    @staticmethod
    def should_retry(
        error_info: ErrorInfo,
        attempt_number: int,
        max_attempts: int,
    ) -> bool:
        """
        Determine if the agent should retry after this failure.
        
        Makes an intelligent decision about retry vs give up:
        - Syntax errors: usually unrecoverable, give up
        - Recoverable errors: retry with reflection
        - Too many attempts: give up to avoid infinite loops
        
        Args:
            error_info: Categorized error
            attempt_number: Current attempt number
            max_attempts: Maximum allowed attempts
        
        Returns:
            True if should retry, False if should give up
        """
        # Don't retry if exceed max attempts
        if attempt_number >= max_attempts:
            logger.warning(f"Max attempts ({max_attempts}) reached. Giving up.")
            return False
        
        # Don't retry syntax errors
        if not error_info.is_recoverable:
            logger.warning(
                f"Error {error_info.error_type} is not recoverable. Giving up."
            )
            return False
        
        logger.info(
            f"Error is recoverable. Retrying (attempt {attempt_number + 1}/{max_attempts})"
        )
        return True
