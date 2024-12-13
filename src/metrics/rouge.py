"""ROUGE metric implementation for lexical similarity evaluation."""

from typing import Dict, List, Optional

from metrics.ngramscorer import NgramScorer


class RougeMetric:
    """ROUGE metric for lexical similarity evaluation."""

    _default_ngram_sizes = (1, 2)

    def __init__(self) -> None:
        """Initialize ROUGE metric."""
        self._scorer = NgramScorer()

    def compute_scores(
        self,
        candidate: str,
        reference: str,
        rouge_types: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores for different n-gram sizes.

        Args:
            candidate: Candidate text
            reference: Reference text
            rouge_types: List of n-gram sizes to compute. Defaults to [1, 2]

        Returns:
            Dictionary with ROUGE scores for each n-gram size
        """
        used_rouge_types = list(self._default_ngram_sizes) if rouge_types is None else rouge_types

        candidate_tokens = self._scorer.tokenize(candidate)
        reference_tokens = self._scorer.tokenize(reference)

        return self._compute_scores_for_types(
            candidate_tokens,
            reference_tokens,
            used_rouge_types,
        )

    def _compute_scores_for_types(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str],
        rouge_types: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores for given n-gram sizes.

        Args:
            candidate_tokens: Tokenized candidate text
            reference_tokens: Tokenized reference text
            rouge_types: List of n-gram sizes to compute

        Returns:
            Dictionary with ROUGE scores for each n-gram size
        """
        scores: Dict[str, Dict[str, float]] = {}
        for ngram_size in rouge_types:
            rouge_key = f'rouge-{ngram_size}'
            precision, recall, f1_score = self._scorer.compute_rouge_n(
                candidate_tokens,
                reference_tokens,
                ngram_size,
            )
            scores[rouge_key] = {
                'precision': precision,
                'recall': recall,
                'f1': f1_score,
            }
        return scores
