"""Tests for document chunking functionality."""

from InformationRetrieval.text_parser import DocumentChunker, ParsedText

# Constants for testing
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_AVG_WORDS = 0


def test_document_chunker_initialization():
    """Test document chunker initialization."""
    chunker = DocumentChunker(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        language='russian',
    )
    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE
    assert chunker.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert chunker.language == 'russian'


def test_document_chunker_language_fallback():
    """Test chunker language fallback behavior."""
    chunker = DocumentChunker(language='unknown')
    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE
    assert chunker.chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_chunk_basic_metadata(document_chunker, long_parsed_text):
    """Test basic chunk metadata."""
    chunks = list(document_chunker.create_chunks(long_parsed_text))
    assert chunks  # Ensure there are chunks

    first_chunk = chunks[0]
    assert isinstance(first_chunk, ParsedText)
    assert first_chunk.word_count > 0


def test_chunk_extended_metadata(document_chunker, long_parsed_text):
    """Test extended chunk metadata."""
    chunks = list(document_chunker.create_chunks(long_parsed_text))
    if not chunks:
        return

    first_chunk = chunks[0]
    assert first_chunk.metadata['is_chunk'] is True
    assert 'chunk_start' in first_chunk.metadata
    assert 'chunk_end' in first_chunk.metadata


def test_chunk_overlap(document_chunker, long_parsed_text):
    """Test chunk overlap behavior."""
    chunks = list(document_chunker.create_chunks(long_parsed_text))
    if len(chunks) <= 1:
        return  # Skip overlap test for single chunk

    first_chunk = chunks[0]
    second_chunk = chunks[1]

    # Calculate actual overlap
    overlap = first_chunk.metadata['chunk_end'] - second_chunk.metadata['chunk_start']

    # Verify overlap exists and matches configured overlap
    assert overlap > 0, 'Chunks should overlap'
    assert (
        overlap >= document_chunker.chunk_overlap
    ), f'Overlap should be at least {document_chunker.chunk_overlap} characters'


def test_empty_document_chunking(document_chunker):
    """Test chunking of empty document."""
    empty_text = ParsedText(
        tokens=[],
        word_count=0,
        sentence_count=0,
        metadata={'language': 'en', 'avg_words_per_sentence': DEFAULT_AVG_WORDS},
    )
    chunks = list(document_chunker.create_chunks(empty_text))
    assert not chunks  # Should return no chunks for empty text
