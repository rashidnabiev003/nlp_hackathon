"""Document parsing module for the LLaMA RAG system."""

import logging
from typing import List

import mammoth
import pandas as pd
from bs4 import BeautifulSoup

from RAG.types import DocumentData

logger = logging.getLogger(__name__)


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    """Extract paragraphs from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        List[str]: Extracted paragraphs
    """
    paragraphs = []
    for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        text = elem.get_text(strip=True)
        if text:
            paragraphs.append(text)
    return paragraphs


def extract_tables(soup: BeautifulSoup) -> tuple[List[List[List[str]]], List[pd.DataFrame]]:
    """Extract tables from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        tuple: (raw tables data, pandas dataframes)
    """
    tables_data = []
    dataframes = []

    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        table_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            table_data.append(row_data)

        tables_data.append(table_data)
        if len(table_data) > 1:
            headers = table_data[0]
            df = pd.DataFrame(table_data[1:], columns=headers)
        else:
            df = pd.DataFrame(table_data)
        dataframes.append(df)

    return tables_data, dataframes


def parse_docx(filepath: str) -> DocumentData:
    """Parse DOCX file into structured data.

    Args:
        filepath: Path to DOCX file

    Returns:
        DocumentData: Structured document data
    """
    with open(filepath, 'rb') as docx_file:
        conversion_result = mammoth.convert_to_html(docx_file)
        html_content = conversion_result.value

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = extract_paragraphs(soup)
    tables_data, dataframes = extract_tables(soup)

    return DocumentData(
        paragraphs=paragraphs,
        tables=tables_data,
        dataframes=dataframes,
    )
