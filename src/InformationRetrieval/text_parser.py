import re
from dataclasses import dataclass
from typing import Dict, List, Union

MetadataValue = Union[str, float, int]


@dataclass
class ParsedText:
    """Data class to store parsed text information."""

    tokens: List[str]
    word_count: int
    sentence_count: int
    metadata: Dict[str, MetadataValue]


class TextParser:
    """Text parser for IR tasks with basic preprocessing capabilities."""

    def __init__(self, language: str = 'en'):
        """Initialize parser with specific language.

        Args:
                language: Language code (default: 'en')
        """
        self.language = language
        self._sentence_end_markers = {
            'en': ['.', '!', '?'],
            'ru': ['.', '!', '?'],
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
                text: Input text to tokenize

        Returns:
                List of tokens
        """
        # Basic tokenization - split by whitespace and remove punctuation
        return re.findall(r'\w+', text.lower())

    def count_sentences(self, text: str) -> int:
        """Count number of sentences in text.

        Args:
                text: Input text

        Returns:
                Number of sentences
        """
        markers = self._sentence_end_markers.get(self.language, ['.', '!', '?'])
        count = sum(1 for char in text if char in markers)
        return max(1, count)  # At least 1 sentence

    def parse(self, text: str) -> ParsedText:
        """Parse text and return structured information.

        Args:
                text: Input text to parse

        Returns:
                ParsedText object with parsing results
        """
        tokens = self.tokenize(text)
        sentences = self.count_sentences(text)

        avg_words = len(tokens) / sentences if sentences > 0 else 0
        return ParsedText(
            tokens=tokens,
            word_count=len(tokens),
            sentence_count=sentences,
            metadata={
                'language': self.language,
                'avg_words_per_sentence': avg_words,
            },
        )
