"""HTML processing utilities for the LLaMA RAG system."""

from typing import List, NamedTuple

import pandas as pd
from bs4 import BeautifulSoup


class TableData(NamedTuple):
    """Structure for table data."""

    raw_data: List[List[List[str]]]
    dataframes: List[pd.DataFrame]


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


def process_table(table_element: BeautifulSoup) -> tuple[List[List[str]], pd.DataFrame]:
    """Process a single table element.

    Args:
        table_element: BeautifulSoup table element

    Returns:
        tuple: (raw table data, pandas dataframe)
    """
    rows = table_element.find_all('tr')
    table_data = [[cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])] for row in rows]

    if len(table_data) > 1:
        headers = table_data[0]
        df = pd.DataFrame(table_data[1:], columns=headers)
    else:
        df = pd.DataFrame(table_data)

    return table_data, df


def extract_tables(soup: BeautifulSoup) -> TableData:
    """Extract tables from HTML soup.

    Args:
        soup: BeautifulSoup object containing HTML

    Returns:
        TableData: Extracted table data
    """
    tables_data = []
    dataframes = []

    for table in soup.find_all('table'):
        table_data, df = process_table(table)
        tables_data.append(table_data)
        dataframes.append(df)

    return TableData(tables_data, dataframes)
