import numpy as np
import pytest

from InformationRetrieval.text_embedder import TextEmbedder
from tests.InformationRetrieval.conftest import TextEmbedderData


@pytest.fixture(name='embedder_fixture')
def fixture_embedder_fixture():
    return TextEmbedder(
        embedding_dim=TextEmbedderData.embedding_dim,
        random_seed=TextEmbedderData.random_seed,
    )


class TestTextEmbedder:
    """Tests for TextEmbedder class."""

    def test_embed_word(self, embedder_fixture):
        """Test single word embedding."""
        vector = embedder_fixture.embed_word(TextEmbedderData.word)

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (TextEmbedderData.embedding_dim,)

        # Test consistency - same word should get same vector
        vector2 = embedder_fixture.embed_word(TextEmbedderData.word)
        np.testing.assert_array_equal(vector, vector2)

    def test_embed_word_case_insensitive(self, embedder_fixture):
        """Test case insensitivity in word embedding."""
        vector1 = embedder_fixture.embed_word(TextEmbedderData.word.title())
        vector2 = embedder_fixture.embed_word(TextEmbedderData.word)

        np.testing.assert_array_equal(vector1, vector2)

    def test_embed_text_string(self, embedder_fixture):
        """Test text embedding from string input."""
        vector = embedder_fixture.embed_text(TextEmbedderData.sentence)

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (TextEmbedderData.embedding_dim,)

    def test_embed_text_tokens(self, embedder_fixture):
        """Test text embedding from token list."""
        vector = embedder_fixture.embed_text(TextEmbedderData.tokens)

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (TextEmbedderData.embedding_dim,)

    def test_embed_text_empty(self, embedder_fixture):
        """Test embedding of empty text."""
        vector = embedder_fixture.embed_text(TextEmbedderData.empty)

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (TextEmbedderData.embedding_dim,)
        np.testing.assert_array_equal(vector, np.zeros(TextEmbedderData.embedding_dim))

    def test_embedding_deterministic(self):
        """Test that embeddings are deterministic with same seed."""
        embedder1 = TextEmbedder(
            embedding_dim=TextEmbedderData.embedding_dim,
            random_seed=TextEmbedderData.random_seed,
        )
        embedder2 = TextEmbedder(
            embedding_dim=TextEmbedderData.embedding_dim,
            random_seed=TextEmbedderData.random_seed,
        )

        vector1 = embedder1.embed_text(TextEmbedderData.word)
        vector2 = embedder2.embed_text(TextEmbedderData.word)

        np.testing.assert_array_equal(vector1, vector2)
