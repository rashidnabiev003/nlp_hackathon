"""HTML processing utilities for the LLaMA RAG system."""

from typing import List

import pandas as pd
from bs4 import BeautifulSoup

from RAG.types import TableData as RawTableData
from RAG.types import TableExtractionResult, TableRow

# HTML tags for text extraction
TEXT_TAGS = ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li')
TABLE_CELL_TAGS = ('td', 'th')


def _extract_text(element: BeautifulSoup) -> str:
    """Extract text from a BeautifulSoup element.

    Args:
        element: BeautifulSoup element

    Returns:
        str: Extracted text
    """
    return element.get_text(strip=True)


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    """Extract paragraphs from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        List[str]: Extracted paragraphs
    """
    elements = soup.find_all(TEXT_TAGS)
    texts = [_extract_text(element) for element in elements]
    return [text for text in texts if text]


def _process_row(row: BeautifulSoup) -> TableRow:
    """Process a single table row.

    Args:
        row: BeautifulSoup row element

    Returns:
        TableRow: Processed row data
    """
    cells = row.find_all(TABLE_CELL_TAGS)
    return [_extract_text(cell) for cell in cells]


def _create_dataframe(table_data: RawTableData) -> pd.DataFrame:
    """Create a DataFrame from table data.

    Args:
        table_data: Raw table data

    Returns:
        pd.DataFrame: Created DataFrame
    """
    if len(table_data) > 1:
        return pd.DataFrame(table_data[1:], columns=table_data[0])
    return pd.DataFrame(table_data)


def process_table(table_element: BeautifulSoup) -> tuple[RawTableData, pd.DataFrame]:
    """Process a single table element.

    Args:
        table_element: BeautifulSoup table element

    Returns:
        tuple: (raw table data, pandas dataframe)
    """
    rows = table_element.find_all('tr')
    table_data = [_process_row(row) for row in rows]
    return table_data, _create_dataframe(table_data)


def extract_tables(soup: BeautifulSoup) -> TableExtractionResult:
    """Extract tables from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        TableExtractionResult: Extracted table data
    """
    tables_data = []
    dataframes = []

    for table in soup.find_all('table'):
        table_data, df = process_table(table)
        tables_data.append(table_data)
        dataframes.append(df)

    return tables_data, dataframes
