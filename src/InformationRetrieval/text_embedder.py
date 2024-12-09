from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray


class TextEmbedder:
    """Basic text embedder that converts text into vector representations."""

    def __init__(self, embedding_dim: int = 100, random_seed: int = 42):
        """Initialize text embedder.

        Args:
                embedding_dim: Dimension of the embedding vectors
                random_seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self._rng = np.random.Generator(np.random.PCG64(random_seed))
        self._word_vectors: Dict[str, NDArray] = {}

    def embed_word(self, word: str) -> NDArray:
        """Create embedding for a single word.

        Args:
                word: Input word

        Returns:
                Word embedding as numpy array
        """
        return self._get_word_vector(word.lower())

    def embed_text(self, text: Union[str, List[str]]) -> NDArray:
        """Create embedding for text or list of tokens.

        Args:
                text: Input text or list of tokens

        Returns:
                Text embedding as numpy array
        """
        if isinstance(text, str):
            # Simple tokenization for demonstration
            tokens = text.lower().split()
        else:
            tokens = [token.lower() for token in text]

        if not tokens:
            return np.zeros(self.embedding_dim)

        # Average word vectors for text embedding
        vectors = []
        for token in tokens:
            vectors.append(self._get_word_vector(token))
        return np.mean(vectors, axis=0)

    def _get_word_vector(self, word: str) -> NDArray:
        """Get vector for a single word, create if not exists.

        Args:
                word: Input word

        Returns:
                Word vector as numpy array
        """
        if word not in self._word_vectors:
            # Create a random vector for demonstration
            # In real implementation, this would use a pre-trained model
            vector = self._rng.normal(0, 0.1, self.embedding_dim)
            self._word_vectors[word] = vector
        return self._word_vectors[word]
