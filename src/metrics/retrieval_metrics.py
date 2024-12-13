"""Retrieval-specific metrics implementation."""

from typing import Optional, Sequence, Union

import numpy as np

Number = Union[int, float]

GAIN_METHOD_STANDARD = 'standard'
GAIN_METHOD_EXPONENTIAL = 'exponential'


def _compute_gain(relevance: Number, method: str = GAIN_METHOD_STANDARD) -> float:
    """Compute gain for a relevance score.

    Args:
        relevance: Relevance score
        method: Gain calculation method ('standard' or 'exponential')

    Returns:
        Computed gain value
    """
    if method == GAIN_METHOD_EXPONENTIAL:
        return float(2**relevance - 1)
    return float(relevance)


def _compute_dcg(
    relevance_scores: Sequence[Number],
    method: str = GAIN_METHOD_STANDARD,
) -> float:
    """Compute Discounted Cumulative Gain (DCG).

    Args:
        relevance_scores: List of relevance scores
        method: Gain calculation method ('standard' or 'exponential')

    Returns:
        DCG score
    """
    if not relevance_scores:
        return 0

    dcg_value = 0
    for position, relevance in enumerate(relevance_scores, 1):
        gain = _compute_gain(relevance, method)
        discount = np.log2(position + 1)
        dcg_value += gain / discount

    return dcg_value


class RetrievalMetrics:
    """Collection of retrieval-specific metrics."""

    def compute_mrr(
        self,
        relevance_lists: Sequence[Sequence[Number]],
        top_limit: Optional[int] = None,
    ) -> float:
        """Compute average MRR over multiple queries.

        Args:
            relevance_lists: List of relevance score lists
            top_limit: Consider only top-k results

        Returns:
            Average MRR score
        """
        if not relevance_lists:
            return 0

        mrr_scores = [self._compute_single_mrr(scores, top_limit) for scores in relevance_lists]
        return float(np.mean(mrr_scores))

    def compute_ndcg(
        self,
        relevance_lists: Sequence[Sequence[Number]],
        top_limit: Optional[int] = None,
        method: str = GAIN_METHOD_STANDARD,
    ) -> float:
        """Compute average NDCG over multiple queries.

        Args:
            relevance_lists: List of relevance score lists
            top_limit: Calculate NDCG@k
            method: Gain calculation method ('standard' or 'exponential')

        Returns:
            Average NDCG score
        """
        if not relevance_lists:
            return 0

        ndcg_scores = [self._compute_single_ndcg(scores, top_limit, method) for scores in relevance_lists]
        return float(np.mean(ndcg_scores))

    def _compute_single_mrr(
        self,
        relevance_scores: Sequence[Number],
        top_limit: Optional[int] = None,
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR) for a single query.

        Args:
            relevance_scores: List of relevance scores
            top_limit: Consider only top-k results

        Returns:
            MRR score
        """
        if not relevance_scores:
            return float(0)

        scores_to_use = relevance_scores[:top_limit] if top_limit is not None else relevance_scores

        reciprocal_rank = float(0)
        for position, score in enumerate(scores_to_use, 1):
            if score > 0:
                reciprocal_rank = 1.0 / position
                break

        return reciprocal_rank

    def _compute_single_ndcg(
        self,
        relevance_scores: Sequence[Number],
        top_limit: Optional[int] = None,
        method: str = GAIN_METHOD_STANDARD,
    ) -> float:
        """Calculate NDCG for a single query.

        Args:
            relevance_scores: List of relevance scores
            top_limit: Calculate NDCG@k
            method: Gain calculation method ('standard' or 'exponential')

        Returns:
            NDCG score
        """
        if not relevance_scores:
            return float(0)

        scores_to_use = relevance_scores[:top_limit] if top_limit is not None else relevance_scores

        dcg = _compute_dcg(scores_to_use, method)
        ideal_scores = sorted(scores_to_use, reverse=True)
        idcg = _compute_dcg(ideal_scores, method)

        return dcg / idcg if idcg > 0 else float(0)
