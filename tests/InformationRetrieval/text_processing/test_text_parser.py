"""Tests for text parser functionality."""

from InformationRetrieval.text_parser import ParsedText, TextParser

# Constants for testing
SINGLE_SENTENCE_WORD_COUNT = 5
SINGLE_SENTENCE_AVG_WORDS = 5.0
FLOAT_COMPARISON_TOLERANCE = 1e-10


def test_text_parser_initialization(text_parser):
    """Test text parser initialization."""
    assert text_parser.language == 'en'
    assert text_parser.lang_map['en'] == 'english'


def test_text_parser_language_mapping():
    """Test language mapping functionality."""
    parser = TextParser(language='ru')
    assert parser.lang_map['ru'] == 'russian'


def test_parse_metadata(text_parser, sample_text):
    """Test parsing metadata."""
    parsed = text_parser.parse(sample_text)
    assert isinstance(parsed, ParsedText)
    assert parsed.metadata['language'] == 'en'
    assert 'avg_words_per_sentence' in parsed.metadata


def test_parse_counts(text_parser, sample_text):
    """Test word and sentence counting."""
    parsed = text_parser.parse(sample_text)
    assert parsed.word_count > 0
    assert parsed.sentence_count > 0
    assert len(parsed.tokens) == parsed.word_count


def test_parse_empty_text(text_parser):
    """Test parsing empty text."""
    parsed = text_parser.parse('')
    assert parsed.word_count == 0
    assert parsed.sentence_count == 0
    assert parsed.metadata['avg_words_per_sentence'] == 0


def test_parse_single_sentence(text_parser):
    """Test parsing single sentence."""
    text = 'This is a single sentence.'
    parsed = text_parser.parse(text)
    assert parsed.sentence_count == 1
    assert parsed.word_count == SINGLE_SENTENCE_WORD_COUNT
    assert abs(parsed.metadata['avg_words_per_sentence'] - SINGLE_SENTENCE_AVG_WORDS) < FLOAT_COMPARISON_TOLERANCE
