"""Fixtures for chunking tests."""

import pytest

from InformationRetrieval.text_parser import DocumentChunker, ParsedText

# Constants
DEFAULT_SENTENCE_COUNT = 20
DEFAULT_WORD_COUNT = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


@pytest.fixture
def sample_parsed_text() -> ParsedText:
    """Create a sample parsed text for testing.

    Returns:
        ParsedText: Sample text with metadata
    """
    # Create a list of words
    words = [f'word{count}' for count in range(DEFAULT_WORD_COUNT)]

    return ParsedText(
        tokens=words,
        word_count=len(words),
        sentence_count=1,
        metadata={
            'language': 'en',
            'avg_words_per_sentence': float(len(words)),
        },
    )


@pytest.fixture
def long_parsed_text() -> ParsedText:
    """Create a longer parsed text for testing.

    Returns:
        ParsedText: Long text with multiple sentences
    """
    # Create sentences with incrementing numbers
    sentences = [f'This is sentence number {count}.' for count in range(DEFAULT_SENTENCE_COUNT)]
    text = ' '.join(sentences)
    words = text.split()

    return ParsedText(
        tokens=words,
        word_count=len(words),
        sentence_count=DEFAULT_SENTENCE_COUNT,
        metadata={
            'language': 'en',
            'avg_words_per_sentence': float(len(words)) / DEFAULT_SENTENCE_COUNT,
        },
    )


@pytest.fixture
def document_chunker():
    """Create document chunker instance for testing.

    Returns:
        DocumentChunker: Configured document chunker instance
    """
    return DocumentChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        language='english',
    )
