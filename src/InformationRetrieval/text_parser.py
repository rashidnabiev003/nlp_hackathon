import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

MetadataValue = Union[str, float, int]

logger = logging.getLogger(__name__)


@dataclass
class ParsedText:
    """Data class to store parsed text information."""

    tokens: List[str]
    word_count: int
    sentence_count: int
    metadata: Dict[str, MetadataValue]


class TextParser:
    """Text parser using LangChain for IR tasks."""

    def __init__(self, language: str = 'en'):
        """Initialize parser with specific language.

        Args:
            language: Language code (default: 'en')
        """
        self.language = language
        # Map common language codes
        self.lang_map = {
            'en': 'english',
            'ru': 'russian',
            # Add more languages as needed
        }

        # Use regular RecursiveCharacterTextSplitter for natural language
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Default chunk size
            chunk_overlap=0,  # No overlap for basic parsing
            separators=['\n\n', '\n', '.', '!', '?', ' ', ''],  # Common natural language separators
        )

    def parse(self, text: str) -> ParsedText:
        """Parse text and return structured information.

        Args:
            text: Input text to parse

        Returns:
            ParsedText object with parsing results
        """
        # Split into chunks (sentences in this case)
        chunks = self.text_splitter.split_text(text)

        # Get tokens (words)
        tokens = text.split()
        token_count = len(tokens)
        chunk_count = len(chunks)

        return ParsedText(
            tokens=tokens,
            word_count=token_count,
            sentence_count=chunk_count,
            metadata={
                'language': self.language,
                'avg_words_per_sentence': token_count / chunk_count if chunk_count else 0,
            },
        )


class DocumentChunker:
    """Handles document chunking using LangChain's text splitter."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        language: str = 'russian',
    ):
        """Initialize chunker with configuration.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            language: Language code ('russian' or 'english')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language

        # Use regular RecursiveCharacterTextSplitter for natural language
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '.', '!', '?', ' ', ''],  # Common natural language separators
        )

    def create_chunks(self, parsed_text: ParsedText) -> Iterator[ParsedText]:
        """Create overlapping chunks from parsed text.

        Args:
            parsed_text: The text to chunk

        Yields:
            ParsedText: Each chunk of the text as a ParsedText object
        """
        text = ' '.join(parsed_text.tokens)
        chunks = self.text_splitter.split_text(text)
        position = 0

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_data = self._process_chunk(
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                position=position,
                parsed_text=parsed_text,
            )
            position = int(chunk_data.metadata['chunk_end'])
            yield chunk_data

    def _process_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        position: int,
        parsed_text: ParsedText,
    ) -> ParsedText:
        """Process a single chunk of text.

        Args:
            chunk_text: The chunk text to process
            chunk_index: Index of the current chunk
            position: Current position in text
            parsed_text: Original parsed text for metadata

        Returns:
            ParsedText: Processed chunk
        """
        chunk_tokens = chunk_text.split()
        overlap_size = self.chunk_overlap if chunk_index > 0 else 0
        chunk_start = max(0, position - overlap_size)
        chunk_end = position + len(chunk_text)

        return ParsedText(
            tokens=chunk_tokens,
            word_count=len(chunk_tokens),
            sentence_count=len(chunk_text.split('.')),
            metadata={
                **parsed_text.metadata,
                'chunk_start': int(chunk_start),
                'chunk_end': int(chunk_end),
                'is_chunk': True,
            },
        )
