"""Root conftest for pytest configuration."""

import logging
import os
import sys
from pathlib import Path

import nltk

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SENTENCE_COUNT = 50


def pytest_sessionstart(session):
    """Configure test environment before session starts."""
    # Add src to Python path
    src_path = str(Path(__file__).parent.parent / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Configure tokenizer parallelism
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Download NLTK resources
    try:
        nltk.download('punkt')
    except Exception as error:
        logger.warning('Failed to download NLTK resources: %s', error)
