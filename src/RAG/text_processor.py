"""Text processing utilities for the LLaMA RAG system."""

from collections.abc import Sequence
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer

from RAG.config import CHUNK_SIZE, OVERLAP_SIZE
from RAG.table_formatter import process_tables
from RAG.types import DocumentData

# Initialize resources
nltk.download('punkt')
morph = MorphAnalyzer()


def preprocess_russian_text(text: str) -> str:
    """Lemmatize Russian text using pymorphy2.

    Args:
        text: Input text string to be preprocessed

    Returns:
        str: Preprocessed and lemmatized text with normalized word forms
    """
    tokens = word_tokenize(text, language='russian')
    lemmas = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmas)


def process_text_line(line: str, lemmatize: bool) -> str:
    """Process a single line of text.

    Args:
        line: Text line to process
        lemmatize: Whether to apply lemmatization

    Returns:
        str: Processed text line
    """
    stripped = line.strip()
    return preprocess_russian_text(stripped) if lemmatize and stripped else stripped


def process_paragraphs(paragraphs: List[str], lemmatize: bool) -> List[str]:
    """Process paragraphs into text chunks.

    Args:
        paragraphs: List of paragraphs to process
        lemmatize: Whether to apply lemmatization

    Returns:
        List[str]: Processed text chunks
    """
    processed = []
    for paragraph in paragraphs:
        processed_text = process_text_line(paragraph, lemmatize)
        if processed_text:
            processed.append(processed_text)
    return processed


def process_text_chunks(doc_data: DocumentData, lemmatize: bool = False) -> List[str]:
    """Process document data into text chunks.

    Args:
        doc_data: Document data to process
        lemmatize: Whether to apply lemmatization

    Returns:
        List[str]: Processed text chunks
    """
    text_elements = []

    # Process paragraphs
    text_elements.extend(process_paragraphs(doc_data['paragraphs'], lemmatize))

    # Process tables
    text_elements.extend(
        process_tables(
            doc_data,
            lambda text: process_text_line(text, lemmatize),
        ),
    )

    return chunk_text(text_elements)


def chunk_text(
    text_lines: Sequence[str],
    chunk_size: int = CHUNK_SIZE,
    overlap_size: int = OVERLAP_SIZE,
) -> List[str]:
    """Split text lines into overlapping chunks.

    Args:
        text_lines: Sequence of text lines to process
        chunk_size: Maximum size of each chunk
        overlap_size: Size of overlap between chunks

    Returns:
        List[str]: List of processed text chunks with specified overlap
    """
    chunks = []
    current_chunk = []
    chunk_length = 0

    for line in text_lines:
        line_length = len(line)
        if chunk_length + line_length <= chunk_size:
            current_chunk.append(line)
            chunk_length += line_length
            continue

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            if overlap_size > 0:
                current_chunk = current_chunk[-overlap_size:]
                chunk_length = sum(len(line) for line in current_chunk)
            else:
                current_chunk = []
                chunk_length = 0

        current_chunk.append(line)
        chunk_length += line_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
