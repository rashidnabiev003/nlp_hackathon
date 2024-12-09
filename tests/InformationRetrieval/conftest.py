"""Test configuration and shared fixtures for InformationRetrieval tests."""

from dataclasses import dataclass
from typing import Final, Tuple


@dataclass(frozen=True)
class TextParserData:
    """Test data constants for text parser."""

    sentence: Final[str] = 'Hello, world! This is a test.'
    tokens: Final[Tuple[str, ...]] = (
        'hello',
        'world',
        'this',
        'is',
        'a',
        'test',
    )
    empty: Final[str] = ''
    language: Final[str] = 'en'
    avg_words: Final[float] = 3.0


@dataclass(frozen=True)
class RussianTextData:
    """Russian language test data."""

    sentence: Final[str] = 'Привет, мир! Это тест.'
    tokens: Final[Tuple[str, ...]] = (
        'привет',
        'мир',
        'это',
        'тест',
    )


@dataclass(frozen=True)
class TextEmbedderData:
    """Test data constants for text embedder."""

    word: Final[str] = 'test'
    sentence: Final[str] = 'hello world'
    tokens: Final[Tuple[str, ...]] = ('hello', 'world')
    empty: Final[str] = ''
    embedding_dim: Final[int] = 50
    random_seed: Final[int] = 42
