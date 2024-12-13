"""Model management utilities for the LLaMA RAG system."""

import logging
import os

import torch
from huggingface_hub import login
from langchain import chains, embeddings, llms, prompts, vectorstores
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from RAG.config import MODEL_CONFIG, PROMPT_PARTS

logger = logging.getLogger(__name__)


def initialize_huggingface() -> None:
    """Initialize HuggingFace environment.

    Raises:
        ValueError: If HUGGINGFACE_TOKEN is not set
    """
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise ValueError('HUGGINGFACE_TOKEN environment variable not set')

    login(token=token)
    torch.cuda.empty_cache()


def create_model_pipeline() -> tuple[AutoTokenizer, AutoModelForCausalLM, pipeline]:
    """Create model pipeline.

    Returns:
        tuple: (tokenizer, model, pipeline)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG['name'],
        cache_dir=MODEL_CONFIG['cache_dir'],
    )
    if 'pad_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG['name'],
        device_map='auto',
        cache_dir=MODEL_CONFIG['cache_dir'],
        quantization_config=bnb_config,
    ).to('cuda')

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


def create_qa_chain(text_chunks: list[str]) -> chains.RetrievalQA:
    """Create QA chain with vector store.

    Args:
        text_chunks: Processed text chunks

    Returns:
        RetrievalQA: Configured QA chain
    """
    model_embeddings = embeddings.HuggingFaceEmbeddings(
        model_name='sentence-transformers/distiluse-base-multilingual-cased-v2',
    )
    vectorstore = vectorstores.FAISS.from_texts(text_chunks, model_embeddings)

    prompt = prompts.PromptTemplate(
        input_variables=['context', 'question'],
        template='\n'.join(PROMPT_PARTS),
    )

    _, _, llm_pipeline = create_model_pipeline()
    llm = llms.HuggingFacePipeline(pipeline=llm_pipeline)

    return chains.RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt},
    )
