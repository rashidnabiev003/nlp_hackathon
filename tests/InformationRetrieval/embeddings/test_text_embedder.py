import numpy as np
import pytest

from InformationRetrieval.text_embedder import RussianTransformerEmbedder

MINILM_L6_V2_EMBEDDING_DIMENSIONS = 384
MINILM_L6_V2_BATCH_SIZE = 32


def test_transformer_embedder_initialization(transformer_embedder):
    assert transformer_embedder.embedding_dim == MINILM_L6_V2_EMBEDDING_DIMENSIONS
    assert transformer_embedder.batch_size == MINILM_L6_V2_BATCH_SIZE


def test_embed_single_text(transformer_embedder, sample_parsed_text):
    embeddings = transformer_embedder.embed(sample_parsed_text)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == transformer_embedder.embedding_dim


def test_embed_multiple_texts(transformer_embedder, sample_parsed_text):
    texts = [sample_parsed_text, sample_parsed_text]
    embeddings = transformer_embedder.embed(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, transformer_embedder.embedding_dim)


def test_russian_embedder_initialization():
    with pytest.raises(ValueError):
        RussianTransformerEmbedder(model_type='invalid')

    embedder = RussianTransformerEmbedder(model_type='deeppavlov')
    assert embedder.model_mapping['deeppavlov'] == 'DeepPavlov/rubert-base-cased-sentence'
