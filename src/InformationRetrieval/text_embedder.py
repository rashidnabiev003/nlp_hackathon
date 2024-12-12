from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from InformationRetrieval.text_parser import ParsedText

# Module-level constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 42

# Model mappings
RUSSIAN_MODEL_MAPPING = (
    ('deeppavlov', 'DeepPavlov/rubert-base-cased-sentence'),
    ('sbert', 'sberbank-ai/sbert_large_nlu_ru'),
    ('labse', 'sentence-transformers/LaBSE'),
)


class TextEmbedder(ABC):
    """Abstract base class for text embedding."""

    @abstractmethod
    def embed(self, text: Union[ParsedText, List[ParsedText]]) -> np.ndarray:
        """Convert text into vector embeddings."""
        raise NotImplementedError


class TransformerEmbedder(TextEmbedder):
    """Text embedder using Sentence Transformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedder with specific transformer model.

        Args:
            model_name: Name of the pre-trained model to use
                Default is 'all-MiniLM-L6-v2' which provides good balance of speed and quality
                Other options:
                - 'all-mpnet-base-v2' (higher quality, slower)
                - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
                - 'all-distilroberta-v1' (faster, slightly lower quality)
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = DEFAULT_BATCH_SIZE

    def embed(self, text: Union[ParsedText, List[ParsedText]]) -> np.ndarray:
        """Create embeddings using transformer model.

        Args:
            text: ParsedText object or list of ParsedText objects

        Returns:
            numpy array of embeddings with shape (n_chunks, embedding_dim)
        """
        if isinstance(text, ParsedText):
            text = [text]

        # Reconstruct sentences from tokens for better semantic understanding
        sentences = [' '.join(chunk.tokens) for chunk in text]

        return self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize embeddings
        )


class MultilingualTransformerEmbedder(TransformerEmbedder):
    """Multilingual version of transformer embedder."""

    def __init__(self):
        """Initialize with multilingual model."""
        super().__init__(model_name='paraphrase-multilingual-MiniLM-L12-v2')


class HighQualityTransformerEmbedder(TransformerEmbedder):
    """High quality version of transformer embedder."""

    def __init__(self):
        """Initialize with high quality model."""
        super().__init__(model_name='all-mpnet-base-v2')


class RussianTransformerEmbedder(TransformerEmbedder):
    """Text embedder optimized for Russian language."""

    def __init__(self, model_type: str = 'deeppavlov'):
        """Initialize embedder with Russian-optimized model.

        Args:
            model_type: Type of model to use:
                - 'deeppavlov': RuBERT от DeepPavlov (лучшее качество)
                - 'sbert': SBERT от Sberbank-AI (быстрее)
                - 'labse': Language-agnostic BERT (хорош для многоязычных текстов)

        Raises:
            ValueError: If provided model_type is not supported
        """
        self.model_mapping = dict(RUSSIAN_MODEL_MAPPING)  # For backward compatibility
        if model_type not in self.model_mapping:
            available_types = list(self.model_mapping.keys())
            raise ValueError(
                'Unknown model type: {0}. Available types: {1}'.format(
                    model_type,
                    available_types,
                ),
            )

        super().__init__(model_name=self.model_mapping[model_type])
