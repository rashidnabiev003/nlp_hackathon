import pytest

from InformationRetrieval.text_parser import ParsedText, TextParser


@pytest.fixture
def text_parser():
    return TextParser(language='en')


@pytest.fixture
def sample_text():
    return 'This is a test sentence. Another sentence here! And a third one?'


@pytest.fixture
def parsed_text():
    return ParsedText(
        tokens=['this', 'is', 'a', 'test', 'sentence'],
        word_count=5,
        sentence_count=1,
        metadata={'language': 'en', 'avg_words_per_sentence': 5.0},
    )
