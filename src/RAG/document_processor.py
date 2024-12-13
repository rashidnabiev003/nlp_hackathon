"""Document processing module for the LLaMA RAG system."""

import logging
import os
from typing import List

import mammoth
import torch
from bs4 import BeautifulSoup
from huggingface_hub import login
from langchain import embeddings, llms, vectorstores

from Parsers.llama_parser import parse_md, parse_txt
from RAG import html_processor, model, text_processor, types

logger = logging.getLogger(__name__)


def validate_file(file_path: str, output_format: str) -> None:
    """Validate input file and format.

    Args:
        file_path: Path to the input file
        output_format: Desired output format

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the output format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    if output_format not in {'txt', 'md'}:
        raise ValueError("Output format must be 'txt' or 'md'")


def parse_document(file_path: str, output_format: str) -> List[str]:
    """Parse document and extract text chunks.

    Args:
        file_path: Path to the input file
        output_format: Desired output format

    Returns:
        List[str]: Extracted text chunks
    """
    parser = parse_txt if output_format == 'txt' else parse_md
    parsed_file_path = parser(file_path)

    with open(parsed_file_path, 'r', encoding='utf-8') as doc_file:
        return text_processor.chunk_text(doc_file.readlines())


def initialize_model() -> llms.HuggingFacePipeline:
    """Initialize the language model.

    Returns:
        HuggingFacePipeline: Initialized language model

    Raises:
        ValueError: If HUGGINGFACE_TOKEN is not set
    """
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
    if not huggingface_token:
        raise ValueError('HUGGINGFACE_TOKEN environment variable not set')

    login(token=huggingface_token)
    torch.cuda.empty_cache()

    _, _, llm_pipeline = model.get_model_pipeline()
    return llms.HuggingFacePipeline(pipeline=llm_pipeline)


def create_vectorstore(text_chunks: List[str]) -> vectorstores.FAISS:
    """Create a vector store from text chunks.

    Args:
        text_chunks: List of text chunks to index

    Returns:
        FAISS: Initialized vector store
    """
    model_embeddings = embeddings.HuggingFaceEmbeddings(
        model_name='sentence-transformers/distiluse-base-multilingual-cased-v2',
    )
    return vectorstores.FAISS.from_texts(text_chunks, model_embeddings)


def parse_docx(filepath: str) -> types.DocumentData:
    """Parse DOCX file into structured data.

    Args:
        filepath: Path to DOCX file

    Returns:
        DocumentData: Structured document data
    """
    with open(filepath, 'rb') as docx_file:
        html_content = mammoth.convert_to_html(docx_file).value

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = html_processor.extract_paragraphs(soup)
    tables_data, dataframes = html_processor.extract_tables(soup)

    return types.DocumentData(
        paragraphs=paragraphs,
        tables=types.TableList(tables_data),
        dataframes=dataframes,
    )
