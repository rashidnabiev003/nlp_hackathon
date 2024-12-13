"""Main module for standalone LLaMA RAG system."""

import logging
from typing import Optional

from langchain import chains

from RAG.document_parser import parse_docx
from RAG.io_utils import get_user_input
from RAG.model_manager import create_qa_chain
from RAG.table_query import get_table_cell, parse_cell_request
from RAG.text_processor import process_text_chunks
from RAG.types import DocumentData

logger = logging.getLogger(__name__)


def process_query(
    query: str,
    qa_chain: chains.RetrievalQA,
    doc_data: DocumentData,
) -> str:
    """Process user query.

    Args:
        query: User query
        qa_chain: QA chain
        doc_data: Document data

    Returns:
        str: Response to query
    """
    cell_request = parse_cell_request(query)
    if cell_request:
        table_num, row_num, col_name = cell_request
        return get_table_cell(doc_data['dataframes'], table_num, row_num, col_name)

    return qa_chain.run(query).strip()


def handle_query(
    query: Optional[str],
    qa_chain: chains.RetrievalQA,
    doc_data: DocumentData,
) -> bool:
    """Handle a single query.

    Args:
        query: User query
        qa_chain: QA chain
        doc_data: Document data

    Returns:
        bool: True if chat should continue, False otherwise

    Raises:
        Exception: If query processing fails
    """
    if not query:
        logger.info('Chat session ended')
        return False

    try:
        response = process_query(query, qa_chain, doc_data)
    except Exception as error:
        logger.error('Error processing query: %s', error)
        raise

    logger.info('Bot: %s', response)
    return True


def initialize_qa_system(docx_path: str) -> tuple[chains.RetrievalQA, DocumentData]:
    """Initialize QA system.

    Args:
        docx_path: Path to DOCX file

    Returns:
        tuple: (QA chain, Document data)
    """
    doc_data = parse_docx(docx_path)
    text_chunks = process_text_chunks(doc_data)
    qa_chain = create_qa_chain(text_chunks)
    return qa_chain, doc_data


def run_chat_session(qa_chain: chains.RetrievalQA, doc_data: DocumentData) -> None:
    """Run interactive chat session.

    Args:
        qa_chain: QA chain
        doc_data: Document data
    """
    logger.info("Chat session started. Type 'exit' to end.")
    while True:
        query = get_user_input()
        if not handle_query(query, qa_chain, doc_data):
            break
