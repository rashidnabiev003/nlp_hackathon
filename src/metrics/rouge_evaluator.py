"""ROUGE evaluation functionality."""

from typing import Dict, List, Optional, Sequence

import numpy as np

from metrics.rouge import RougeMetric
from metrics.types import RawRougeScoresList, RougeScores


class RougeEvaluator:
    """ROUGE evaluation functionality."""

    _default_rouge_types = (1, 2)

    def __init__(self, rouge_types: Optional[Sequence[int]] = None) -> None:
        """Initialize ROUGE evaluator.

        Args:
            rouge_types: N-gram sizes for ROUGE metric. Defaults to (1, 2)
        """
        self.rouge = RougeMetric()
        self.rouge_types = list(rouge_types or self._default_rouge_types)

    def compute_single_scores(
        self,
        candidate: str,
        reference: str,
    ) -> Dict[str, RougeScores]:
        """Compute ROUGE scores for a single pair of texts.

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            Dictionary with ROUGE scores for each n-gram size
        """
        raw_scores = self.rouge.compute_scores(
            candidate,
            reference,
            self.rouge_types,
        )
        rouge_scores: Dict[str, RougeScores] = {}

        for rouge_type, scores in raw_scores.items():
            rouge_scores[rouge_type] = RougeScores(
                precision=scores['precision'],
                recall=scores['recall'],
                f1=scores['f1'],
            )

        return rouge_scores

    def compute_average_scores(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Dict[str, RougeScores]:
        """Compute average ROUGE scores for multiple pairs of texts.

        Args:
            candidates: List of candidate texts
            references: List of reference texts

        Returns:
            Dictionary with averaged ROUGE scores for each n-gram size
        """
        # Compute scores for each pair
        all_scores = []
        for cand, ref in zip(candidates, references):
            all_scores.append(self.rouge.compute_scores(cand, ref, self.rouge_types))

        # Calculate averages for each ROUGE type
        avg_rouge: Dict[str, RougeScores] = {}
        for rouge_size in self.rouge_types:
            rouge_type = f'rouge-{rouge_size}'
            avg_rouge[rouge_type] = self._compute_averages(
                all_scores,
                rouge_type,
            )

        return avg_rouge

    def _compute_averages(
        self,
        all_scores: RawRougeScoresList,
        rouge_type: str,
    ) -> RougeScores:
        """Compute average ROUGE scores for a specific type.

        Args:
            all_scores: List of ROUGE scores for all pairs
            rouge_type: Type of ROUGE score to average

        Returns:
            Averaged ROUGE scores
        """
        precisions = []
        recalls = []
        f1s = []

        for scores in all_scores:
            score_dict = scores[rouge_type]
            precisions.append(score_dict['precision'])
            recalls.append(score_dict['recall'])
            f1s.append(score_dict['f1'])

        return RougeScores(
            precision=float(np.mean(precisions)),
            recall=float(np.mean(recalls)),
            f1=float(np.mean(f1s)),
        )
