"""Main module for the LLaMA RAG system."""

import logging

from langchain import chains, llms, prompts, vectorstores

from RAG.chat import chat
from RAG.config import PROMPT_PARTS, TOP_K_DOCS
from RAG.document_processor import (
    create_vectorstore,
    initialize_model,
    parse_document,
    validate_file,
)

logger = logging.getLogger(__name__)


def create_qa_chain(
    llm: llms.HuggingFacePipeline,
    vectorstore: vectorstores.FAISS,
) -> chains.RetrievalQA:
    """Create a RetrievalQA chain with the specified prompt template.

    Args:
        llm: Configured language model pipeline
        vectorstore: Initialized FAISS vector store

    Returns:
        RetrievalQA: Configured question-answering chain
    """
    prompt = prompts.PromptTemplate(
        input_variables=['context', 'question'],
        template='\n'.join(PROMPT_PARTS),
    )

    return chains.RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': TOP_K_DOCS}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt},
    )


def process_docx(file_path: str, output_format: str = 'txt') -> None:
    """Process a DOCX file and initialize QA chain.

    This function handles the complete pipeline from document parsing to
    QA chain initialization and chat interface setup.

    Args:
        file_path: Path to the DOCX file to process
        output_format: Output format for parsing ('txt' or 'md')
    """
    validate_file(file_path, output_format)
    text_chunks = parse_document(file_path, output_format)
    llm = initialize_model()
    vectorstore = create_vectorstore(text_chunks)
    qa_chain = create_qa_chain(llm, vectorstore)
    chat(qa_chain)
