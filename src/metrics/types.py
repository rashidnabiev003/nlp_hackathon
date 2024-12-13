"""Type definitions for metrics."""

from typing import Dict, List, TypedDict, Union

Number = Union[int, float]
RelevanceList = List[Number]
RelevanceLists = List[RelevanceList]

RawRougeScores = Dict[str, Dict[str, float]]
RawRougeScoresList = List[RawRougeScores]


class RougeScores(TypedDict):
    """Type for ROUGE scores."""

    precision: float
    recall: float
    f1: float


class TextSimilarityScores(TypedDict):
    """Type for text similarity scores."""

    bert_score: Union[float, List[float]]
    rouge: Dict[str, RougeScores]


class RetrievalScores(TypedDict):
    """Type for retrieval scores."""

    mrr: float
    ndcg: float
