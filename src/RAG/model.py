"""Model initialization and pipeline setup for the LLaMA RAG system."""

from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from RAG.config import MODEL_CONFIG


def get_model_pipeline(device: str = 'cuda') -> Tuple[AutoTokenizer, AutoModelForCausalLM, pipeline]:
    """Initialize the model pipeline with the specified configuration.

    Args:
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - tokenizer: Configured tokenizer
            - model: Loaded and configured model
            - pipeline: Text generation pipeline
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG['name'],
        cache_dir=MODEL_CONFIG['cache_dir'],
    )

    if 'pad_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG['name'],
        device_map='auto',
        cache_dir=MODEL_CONFIG['cache_dir'],
        quantization_config=bnb_config,
    ).to(device)

    text_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MODEL_CONFIG['max_new_tokens'],
        temperature=MODEL_CONFIG['temperature'],
        top_k=MODEL_CONFIG['top_k'],
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer, model, text_pipeline
