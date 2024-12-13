"""BERTScore implementation for semantic similarity evaluation."""

from typing import List, Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer


class BERTScoreMetric:
    """BERTScore metric for semantic similarity evaluation."""

    _default_model = 'DeepPavlov/rubert-base-cased'
    _max_sequence_length = 512

    def __init__(
        self,
        model_name: str = _default_model,
        device: Optional[str] = None,
    ):
        """Initialize BERTScore metric.

        Args:
            model_name: Name of the BERT model to use
            device: Device to use for computation. Defaults to CUDA if available, else CPU
        """
        self._cached_device: Optional[str] = None
        self.device = device or self._get_default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_score(
        self,
        candidates: Union[str, List[str]],
        references: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """Compute BERTScore between candidate and reference texts.

        Args:
            candidates: Candidate text(s)
            references: Reference text(s)

        Returns:
            BERTScore(s) between candidate and reference texts

        Raises:
            ValueError: If candidates and references are not both strings or both lists,
                       or if they are lists of different lengths
        """
        # Handle single strings
        if isinstance(candidates, str) and isinstance(references, str):
            return self._compute_single_score(candidates, references)

        # Handle lists of strings
        if not isinstance(candidates, list) or not isinstance(references, list):
            raise ValueError('Candidates and references must be both strings or both lists')

        if len(candidates) != len(references):
            raise ValueError('Candidates and references must have the same length')

        computed_scores = []
        for cand, ref in zip(candidates, references):
            computed_scores.append(self._compute_single_score(cand, ref))
        return computed_scores

    def _compute_single_score(self, candidate: str, reference: str) -> float:
        """Compute BERTScore for a single pair of texts.

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            BERTScore between candidate and reference
        """
        cand_embeddings = self._get_embeddings(candidate)
        ref_embeddings = self._get_embeddings(reference)

        similarity = torch.nn.functional.cosine_similarity(
            cand_embeddings,
            ref_embeddings,
        )

        return float(similarity.mean().item())

    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for text.

        Args:
            text: Input text

        Returns:
            Text embeddings
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self._max_sequence_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def _get_default_device(self) -> str:
        """Get default device for computation.

        Returns:
            Device string ('cuda' if available, else CPU)
        """
        if self._cached_device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._cached_device = device
            return device
        return self._cached_device
