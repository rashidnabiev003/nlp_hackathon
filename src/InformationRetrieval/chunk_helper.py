from typing import List


class ChunkHelper:
    """Helper class for document chunking operations."""

    def __init__(self, language: str):
        """Initialize helper with language.

        Args:
            language: Language code for text operations
        """
        self.language = language

    def get_chunk_end_index(self, tokens: List[str], start_idx: int) -> int:
        """Get end index for chunk starting at start_idx.

        Args:
            tokens: List of tokens
            start_idx: Starting index for the chunk

        Returns:
            int: End index for the chunk
        """
        if not tokens[start_idx:]:
            return start_idx

        return self.find_sentence_boundary(tokens, start_idx, 1)

    def get_next_start_index(self, tokens: List[str], current_start: int) -> int:
        """Get start index for next chunk.

        Args:
            tokens: List of tokens
            current_start: Current chunk start index

        Returns:
            int: Start index for next chunk
        """
        if not tokens[current_start:]:
            return current_start

        return self.find_sentence_boundary(tokens, current_start, -1)

    def find_sentence_boundary(self, tokens: List[str], start_idx: int, direction: int) -> int:
        """Find the nearest sentence boundary.

        Args:
            tokens: List of tokens
            start_idx: Starting index
            direction: 1 for forward search, -1 for backward search

        Returns:
            int: Index of the nearest sentence boundary
        """
        # Simplified logic for demonstration
        return start_idx + 1 if direction > 0 else start_idx - 1
