import numpy as np
import pytest

from InformationRetrieval.text_embedder import TransformerEmbedder
from InformationRetrieval.text_parser import ParsedText

MINILM_L6_V2_EMBEDDING_DIMENSIONS = 384


@pytest.fixture
def transformer_embedder():
    return TransformerEmbedder()


@pytest.fixture
def sample_parsed_text():
    return ParsedText(
        tokens=['this', 'is', 'a', 'test', 'sentence'],
        word_count=5,
        sentence_count=1,
        metadata={'language': 'en', 'avg_words_per_sentence': 5.0},
    )


@pytest.fixture
def sample_embeddings():
    return np.random.rand(1, MINILM_L6_V2_EMBEDDING_DIMENSIONS)
