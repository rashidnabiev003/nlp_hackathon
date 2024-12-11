import pytest

from InformationRetrieval.text_parser import ParsedText, TextParser
from tests.InformationRetrieval.conftest import RussianTextData, TextParserData


@pytest.fixture(name='text_fixture')
def fixture_text_fixture():
    return TextParser()


class TestTextParser:
    """Tests for TextParser class."""

    def test_tokenize_basic(self, text_fixture):
        """Test basic tokenization functionality."""
        tokens = text_fixture.tokenize(TextParserData.sentence)
        assert tuple(tokens) == TextParserData.tokens

    def test_tokenize_empty(self, text_fixture):
        """Test tokenization of empty text."""
        tokens = text_fixture.tokenize(TextParserData.empty)
        assert not tokens

    def test_count_sentences_basic(self, text_fixture):
        """Test basic sentence counting."""
        count = text_fixture.count_sentences(TextParserData.sentence)
        assert count == 2

    def test_count_sentences_empty(self, text_fixture):
        """Test sentence counting for empty text."""
        count = text_fixture.count_sentences(TextParserData.empty)
        assert count == 1  # Minimum is 1

    def test_parse_complete(self, text_fixture):
        """Test complete parsing functionality."""
        parsed_text = text_fixture.parse(TextParserData.sentence)

        assert isinstance(parsed_text, ParsedText)
        assert tuple(parsed_text.tokens) == TextParserData.tokens
        assert parsed_text.word_count == len(TextParserData.tokens)
        assert parsed_text.sentence_count == 2
        assert parsed_text.metadata['language'] == TextParserData.language

    def test_parse_avg_words(self, text_fixture):
        """Test average words per sentence calculation."""
        parsed_text = text_fixture.parse(TextParserData.sentence)
        assert parsed_text.metadata['avg_words_per_sentence'] == TextParserData.avg_words

    def test_russian_support(self):
        """Test parser with Russian language support."""
        russian_parser = TextParser(language='ru')
        parsed_text = russian_parser.parse(RussianTextData.sentence)

        assert tuple(parsed_text.tokens) == RussianTextData.tokens
        assert parsed_text.word_count == len(RussianTextData.tokens)
        assert parsed_text.sentence_count == 2
        assert parsed_text.metadata['language'] == 'ru'
