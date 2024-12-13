"""Configuration settings for the LLaMA RAG system."""

from types import MappingProxyType
from typing import Mapping, Sequence

# Constants
CHUNK_SIZE: int = 500
OVERLAP_SIZE: int = 100
TOP_K_DOCS: int = 5

# Model configuration
MODEL_CONFIG: Mapping[str, str | int | float] = MappingProxyType(
    {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'cache_dir': './models_cache',
        'max_new_tokens': 200,
        'temperature': 0.3,
        'top_k': 100,
    },
)

# Prompt template parts
PROMPT_PARTS: Sequence[str] = (
    'You are an intelligent assistant analyzing a document.',
    'Use ONLY the provided context to answer.',
    "If the context doesn't contain the answer, say: 'Insufficient information.'",
    '',
    'Context:\n{context}\n',
    'Question: {question}',
    'Answer:',
)
