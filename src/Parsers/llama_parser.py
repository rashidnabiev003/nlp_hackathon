import os
from typing import Dict, List

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader
from llama_parse import LlamaParse

nest_asyncio.apply()
load_dotenv()


def _create_parser(result_type: str) -> LlamaParse:
    """Create a LlamaParse instance with common configuration.

    Args:
        result_type: Type of parsing result ('markdown' or 'txt')

    Returns:
        LlamaParse: Configured parser instance
    """
    return LlamaParse(
        result_type=result_type,
        show_progress=True,
        parsing_instruction=None,
        disable_ocr=True,
        disable_image_extraction=True,
        api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
    )


def _format_page_content(page_num: int, document: Document) -> str:
    """Format content for a single page.

    Args:
        page_num: Page number
        document: Document to format

    Returns:
        str: Formatted page content
    """
    page_header = f'### Page {page_num}\n\n'
    page_content = f'{document.text}\n\n'
    return page_header + page_content


def _save_content(doc_content: str, output_path: str) -> str:
    """Save content to file.

    Args:
        doc_content: Content to save
        output_path: Path where to save the content

    Returns:
        str: Path to the saved file
    """
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(doc_content)
    return output_path


def _process_documents(documents: List[Document], output_path: str) -> str:
    """Process documents and save to file.

    Args:
        documents: List of documents to process
        output_path: Path where to save the output file

    Returns:
        str: Path to the saved output file
    """
    content_parts = []
    for page_num, document in enumerate(documents, start=1):
        content_parts.append(_format_page_content(page_num, document))
    all_pages_content = ''.join(content_parts)
    return _save_content(all_pages_content, output_path)


def parse_md(file_path: str) -> str:
    """Parse document to markdown format.

    Args:
        file_path: Path to the document to parse

    Returns:
        str: Path to the output markdown file
    """
    parser = _create_parser('markdown')
    file_extractor: Dict[str, LlamaParse] = {'.docx': parser}

    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return _process_documents(documents, 'all_pages_output.md')


def parse_txt(file_path: str) -> str:
    """Parse document to text format.

    Args:
        file_path: Path to the document to parse

    Returns:
        str: Path to the output text file
    """
    parser = _create_parser('txt')
    file_extractor: Dict[str, LlamaParse] = {'.docx': parser}

    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return _process_documents(documents, 'all_pages_output.txt')
