"""Document parsing module for the LLaMA RAG system."""

import logging
from typing import List

import mammoth
import pandas as pd
from bs4 import BeautifulSoup

from RAG.types import DocumentData, TableExtractionResult, TableList

logger = logging.getLogger(__name__)

# HTML tags for text extraction
TEXT_TAGS = ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li')


def _extract_text_from_element(elem: BeautifulSoup) -> str:
    """Extract text from a BeautifulSoup element.

    Args:
        elem: BeautifulSoup element

    Returns:
        str: Extracted text
    """
    return elem.get_text(strip=True)


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    """Extract paragraphs from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        List[str]: Extracted paragraphs
    """
    elements = soup.find_all(TEXT_TAGS)
    texts = [_extract_text_from_element(elem) for elem in elements]
    return [text for text in texts if text]


def _process_table_row(row: BeautifulSoup) -> List[str]:
    """Process a single table row.

    Args:
        row: BeautifulSoup row element

    Returns:
        List[str]: Row data
    """
    cells = row.find_all(['td', 'th'])
    return [_extract_text_from_element(cell) for cell in cells]


def _create_dataframe(table_data: List[List[str]]) -> pd.DataFrame:
    """Create a DataFrame from table data.

    Args:
        table_data: Raw table data

    Returns:
        pd.DataFrame: Created DataFrame
    """
    if len(table_data) > 1:
        return pd.DataFrame(table_data[1:], columns=table_data[0])
    return pd.DataFrame(table_data)


def extract_tables(soup: BeautifulSoup) -> TableExtractionResult:
    """Extract tables from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        TableExtractionResult: Tuple of (raw tables data, pandas dataframes)
    """
    tables_data = []
    dataframes = []

    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        table_data = [_process_table_row(row) for row in rows]
        tables_data.append(table_data)
        dataframes.append(_create_dataframe(table_data))

    return tables_data, dataframes


def parse_docx(filepath: str) -> DocumentData:
    """Parse DOCX file into structured data.

    Args:
        filepath: Path to DOCX file

    Returns:
        DocumentData: Structured document data
    """
    with open(filepath, 'rb') as docx_file:
        html_content = mammoth.convert_to_html(docx_file).value

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = extract_paragraphs(soup)
    tables_data, dataframes = extract_tables(soup)

    return DocumentData(
        paragraphs=paragraphs,
        tables=TableList(tables_data),
        dataframes=dataframes,
    )
