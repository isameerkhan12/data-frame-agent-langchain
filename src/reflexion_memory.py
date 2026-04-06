"""
Reflexion memory system for learning from failed attempts.

This module implements persistent memory that stores lessons learned from
agent failures. When the agent encounters similar errors in future attempts,
it retrieves and applies these lessons to improve its reasoning.

Key differences from standard ConversationBufferMemory:
- Stores structured failure-lesson pairs (error patterns → solutions)
- Cross-attempt learning (lessons persist across multiple queries)
- Semantic similarity matching to find relevant past lessons
- Explicit error categorization (KeyError, TypeError, etc.)
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A lesson learned from a failed attempt."""
    
    problem: str                    # The category/type of problem encountered
    error_message: str              # The actual error text
    error_type: str                 # Error class (KeyError, TypeError, etc.)
    solution: str                   # How to fix it or avoid it
    timestamp: datetime = field(default_factory=datetime.now)
    attempts_count: int = 1         # How many times this lesson was applied


class ReflexionMemory:
    """
    Persistent memory system that stores and retrieves lessons from failures.
    
    This enables the agent to improve its reasoning across multiple attempts
    by remembering what errors it made and how to avoid them.
    
    Example:
        memory = ReflexionMemory()
        memory.add_lesson(
            problem="column_not_found",
            error_message="KeyError: 'Precipitation'",
            error_type="KeyError",
            solution="Check df.columns first and use lowercase column names"
        )
        
        context = memory.get_context("high_precipitation", top_k=3)
        # Returns relevant lessons as prompt context
    """
    
    def __init__(self, max_lessons: int = 50):
        """
        Initialize the reflexion memory.
        
        Args:
            max_lessons: Maximum number of lessons to store. When exceeded,
                        oldest lessons are removed (FIFO). This prevents
                        unbounded memory growth in long-running sessions.
        """
        self.lessons: list[Lesson] = []
        self.max_lessons = max_lessons
        logger.info(f"ReflexionMemory initialized (max_lessons={max_lessons})")
    
    def add_lesson(
        self,
        problem: str,
        error_message: str,
        error_type: str,
        solution: str,
    ) -> None:
        """
        Store a lesson from a failed attempt.
        
        Args:
            problem: Category of the problem (e.g., "column_not_found", 
                    "type_error", "iteration_limit")
            error_message: The actual error text (e.g., "KeyError: 'Precipitation'")
            error_type: Python error class name (e.g., "KeyError", "TypeError")
            solution: How to fix or avoid this error in future attempts
        """
        lesson = Lesson(
            problem=problem,
            error_message=error_message,
            error_type=error_type,
            solution=solution,
        )
        self.lessons.append(lesson)
        
        # Enforce max_lessons constraint
        if len(self.lessons) > self.max_lessons:
            removed = self.lessons.pop(0)
            logger.debug(f"Removed oldest lesson: {removed.problem}")
        
        logger.info(
            f"Lesson added: {problem} ({error_type}). "
            f"Total lessons: {len(self.lessons)}"
        )
    
    def get_context(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve relevant past lessons as context for the prompt.
        
        This finds lessons most relevant to the current question and
        formats them as additional context the agent can use to improve
        its reasoning.
        
        Args:
            question: The current user question/problem
            top_k: Maximum number of lessons to return
        
        Returns:
            Formatted string of relevant lessons, or empty string if none found.
            Format: "Past lessons:\n- Problem: X\n  Error: Y\n  Solution: Z\n..."
        """
        if not self.lessons:
            return ""
        
        # Simple relevance: check if problem keywords appear in question
        relevant_lessons = []
        for lesson in self.lessons:
            # Case-insensitive keyword matching
            if (lesson.problem.lower() in question.lower() or
                lesson.error_type.lower() in question.lower()):
                relevant_lessons.append(lesson)
        
        if not relevant_lessons:
            return ""
        
        # Take most recent relevant lessons (last-added = most recent)
        relevant_lessons = relevant_lessons[-top_k:]
        
        context = "\n📚 PAST LESSONS (use these to improve your reasoning):\n"
        for i, lesson in enumerate(relevant_lessons, 1):
            context += (
                f"{i}. Problem: {lesson.problem}\n"
                f"   Error Type: {lesson.error_type}\n"
                f"   Original Error: {lesson.error_message[:80]}\n"
                f"   Solution: {lesson.solution}\n"
            )
        
        logger.debug(f"Retrieved {len(relevant_lessons)} lessons for context")
        return context
    
    def get_all_lessons_summary(self) -> str:
        """
        Return a human-readable summary of all stored lessons.
        
        Useful for debugging and understanding agent learning across sessions.
        """
        if not self.lessons:
            return "No lessons stored yet."
        
        summary = f"Reflexion Memory Summary ({len(self.lessons)} lessons):\n"
        for i, lesson in enumerate(self.lessons, 1):
            summary += (
                f"{i}. [{lesson.error_type}] {lesson.problem}\n"
                f"   Solution: {lesson.solution}\n"
            )
        return summary
