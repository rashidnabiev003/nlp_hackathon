"""Main evaluator class that combines all metrics."""

from typing import List, Optional, Sequence, Union

import numpy as np

from metrics.bert_score import BERTScoreMetric
from metrics.retrieval_metrics import RetrievalMetrics
from metrics.rouge_evaluator import RougeEvaluator
from metrics.types import RelevanceLists, RetrievalScores, TextSimilarityScores


class Evaluator:
    """Main evaluator class that combines all metrics."""

    def __init__(
        self,
        bert_model_name: str = 'DeepPavlov/rubert-base-cased',
        rouge_types: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize evaluator with all metrics.

        Args:
            bert_model_name: Name of the BERT model for BERTScore
            rouge_types: N-gram sizes for ROUGE metric. Defaults to (1, 2)
        """
        self.bert_score = BERTScoreMetric(model_name=bert_model_name)
        self.rouge = RougeEvaluator(rouge_types=rouge_types)
        self.retrieval_metrics = RetrievalMetrics()

    def evaluate_text_similarity(
        self,
        candidates: Union[str, List[str]],
        references: Union[str, List[str]],
    ) -> TextSimilarityScores:
        """Evaluate text similarity using BERTScore and ROUGE.

        Args:
            candidates: Candidate text(s)
            references: Reference text(s)

        Returns:
            Dictionary with all text similarity metrics

        Raises:
            ValueError: If candidates and references are not both strings or both lists
        """
        # Compute BERTScore
        bert_scores = self.bert_score.compute_score(candidates, references)

        # Handle single strings
        if self._are_single_texts(candidates, references):
            return self._evaluate_single_texts(
                candidates,  # type: ignore
                references,  # type: ignore
                bert_scores,
            )

        # Handle lists of strings
        if self._are_text_lists(candidates, references):
            return self._evaluate_text_lists(
                candidates,  # type: ignore
                references,  # type: ignore
                bert_scores,
            )

        raise ValueError(
            'Candidates and references must both be strings or both be lists',
        )

    def evaluate_retrieval(
        self,
        relevance_lists: RelevanceLists,
        top_limit: Optional[int] = None,
    ) -> RetrievalScores:
        """Evaluate retrieval performance using MRR and NDCG.

        Args:
            relevance_lists: List of relevance score lists
            top_limit: Consider only top-k results

        Returns:
            Dictionary with retrieval metrics
        """
        mrr = self.retrieval_metrics.compute_mrr(relevance_lists, top_limit)
        ndcg = self.retrieval_metrics.compute_ndcg(relevance_lists, top_limit)

        return RetrievalScores(mrr=mrr, ndcg=ndcg)

    def _are_single_texts(
        self,
        candidates: Union[str, List[str]],
        references: Union[str, List[str]],
    ) -> bool:
        """Check if inputs are single texts.

        Args:
            candidates: Candidate text(s)
            references: Reference text(s)

        Returns:
            True if both inputs are strings
        """
        return isinstance(candidates, str) and isinstance(references, str)

    def _are_text_lists(
        self,
        candidates: Union[str, List[str]],
        references: Union[str, List[str]],
    ) -> bool:
        """Check if inputs are lists of texts.

        Args:
            candidates: Candidate text(s)
            references: Reference text(s)

        Returns:
            True if both inputs are lists
        """
        return isinstance(candidates, list) and isinstance(references, list)

    def _evaluate_single_texts(
        self,
        candidate: str,
        reference: str,
        bert_scores: Union[float, List[float]],
    ) -> TextSimilarityScores:
        """Evaluate similarity for single texts.

        Args:
            candidate: Candidate text
            reference: Reference text
            bert_scores: BERTScore result

        Returns:
            Similarity scores
        """
        rouge_scores = self.rouge.compute_single_scores(candidate, reference)
        return TextSimilarityScores(
            bert_score=bert_scores,
            rouge=rouge_scores,
        )

    def _evaluate_text_lists(
        self,
        candidates: List[str],
        references: List[str],
        bert_scores: Union[float, List[float]],
    ) -> TextSimilarityScores:
        """Evaluate similarity for lists of texts.

        Args:
            candidates: List of candidate texts
            references: List of reference texts
            bert_scores: BERTScore results

        Returns:
            Similarity scores
        """
        rouge_scores = self.rouge.compute_average_scores(candidates, references)
        if isinstance(bert_scores, list):
            avg_bert_score = float(np.mean(bert_scores))
        else:
            avg_bert_score = bert_scores

        return TextSimilarityScores(
            bert_score=avg_bert_score,
            rouge=rouge_scores,
        )
