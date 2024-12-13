from collections import Counter
from typing import List, Tuple

import nltk
from nltk.util import ngrams


class NgramScorer:
    """Helper class for n-gram based scoring operations."""

    def __init__(self) -> None:
        """Initialize NgramScorer."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def compute_rouge_n(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str],
        ngram_size: int,
    ) -> Tuple[float, float, float]:
        """Compute ROUGE-N scores.

        Args:
            candidate_tokens: Candidate tokens
            reference_tokens: Reference tokens
            ngram_size: N-gram size

        Returns:
            Tuple of precision, recall, and F1 scores
        """
        # Get n-grams and counts
        candidate_counts = self._get_ngram_counts(candidate_tokens, ngram_size)
        reference_counts = self._get_ngram_counts(reference_tokens, ngram_size)
        overlap_count = self._compute_overlap(candidate_counts, reference_counts)

        # Calculate metrics
        precision = self._divide_or_zero(overlap_count, sum(candidate_counts.values()))
        recall = self._divide_or_zero(overlap_count, sum(reference_counts.values()))
        return precision, recall, self._compute_f1(precision, recall)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return nltk.word_tokenize(text.lower())

    def _get_ngram_counts(
        self,
        tokens: List[str],
        ngram_size: int,
    ) -> Counter[Tuple[str, ...]]:
        """Get n-grams from tokens.

        Args:
            tokens: List of tokens
            ngram_size: N-gram size

        Returns:
            Counter of n-grams
        """
        ngram_tuples = ngrams(tokens, ngram_size)
        return Counter(tuple(gram) for gram in ngram_tuples)

    def _compute_overlap(
        self,
        counter1: Counter[Tuple[str, ...]],
        counter2: Counter[Tuple[str, ...]],
    ) -> int:
        """Compute overlap between two counters.

        Args:
            counter1: First counter
            counter2: Second counter

        Returns:
            Sum of minimum counts for shared elements
        """
        return sum((counter1 & counter2).values())

    def _divide_or_zero(self, numerator: int, denominator: int) -> float:
        """Safely divide two numbers, returning 0 if denominator is 0.

        Args:
            numerator: Number to divide
            denominator: Number to divide by

        Returns:
            Result of division or 0 if denominator is 0
        """
        return numerator / denominator if denominator > 0 else 0

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall.

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score
        """
        denominator = precision + recall
        denominator_count = 2 * precision * recall
        return 0 if denominator == 0 else denominator_count / denominator
