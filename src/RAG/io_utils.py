"""I/O utilities for the LLaMA RAG system."""

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def get_exit_commands() -> frozenset[str]:
    """Get set of exit commands.

    Returns:
        frozenset[str]: Set of exit commands
    """
    return frozenset(('exit', 'выход'))


def process_input(query: str) -> Optional[str]:
    """Process user input.

    Args:
        query: Raw user input

    Returns:
        Optional[str]: Processed input or None if exit command
    """
    stripped = query.strip()
    return None if stripped.lower() in get_exit_commands() else stripped


def get_user_input(prompt: str = 'Question: ') -> Optional[str]:
    """Get user input safely.

    Args:
        prompt: Input prompt to display

    Returns:
        Optional[str]: User input or None if exit command
    """
    try:
        return process_input(sys.stdin.readline())
    except (KeyboardInterrupt, EOFError):
        return None
