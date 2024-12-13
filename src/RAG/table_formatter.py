"""Table formatting utilities for the LLaMA RAG system."""

from typing import Callable, List, Optional

import pandas as pd

from RAG.types import DocumentData


def format_cell(column: str, cell_content: str) -> str:
    """Format a cell value with its column name.

    Args:
        column: Column name
        cell_content: Cell content

    Returns:
        str: Formatted cell value
    """
    return f'{column}: {cell_content}'


def format_table_prefix(table_idx: int) -> str:
    """Format table prefix.

    Args:
        table_idx: Table index

    Returns:
        str: Formatted table prefix
    """
    return f'Table {table_idx + 1}'


def format_row_prefix(row_idx: int) -> str:
    """Format row prefix.

    Args:
        row_idx: Row index

    Returns:
        str: Formatted row prefix
    """
    return f'Row {row_idx + 1}'


def format_location(table_idx: int, row_idx: int) -> str:
    """Format table and row location.

    Args:
        table_idx: Table index
        row_idx: Row index

    Returns:
        str: Formatted location string
    """
    table_prefix = format_table_prefix(table_idx)
    row_prefix = format_row_prefix(row_idx)
    return f'{table_prefix}, {row_prefix}:'


def process_row_data(row_data: pd.Series) -> List[str]:
    """Process row data into formatted cells.

    Args:
        row_data: Row data

    Returns:
        List[str]: List of formatted cells
    """
    formatted_cells = []
    for column, cell_content in row_data.items():
        if pd.notna(cell_content) and str(cell_content).strip():
            formatted_cells.append(format_cell(column, str(cell_content)))
    return formatted_cells


def format_row(
    table_idx: int,
    row_idx: int,
    row_data: pd.Series,
) -> Optional[str]:
    """Format a table row into a text string.

    Args:
        table_idx: Table index
        row_idx: Row index
        row_data: Row data

    Returns:
        Optional[str]: Formatted row text or None if empty
    """
    formatted_cells = process_row_data(row_data)
    if not formatted_cells:
        return None

    location = format_location(table_idx, row_idx)
    content = ' | '.join(formatted_cells)
    return f'{location} {content}'


def process_table_row(
    table_idx: int,
    row_idx: int,
    row: pd.Series,
    process_line: Callable[[str], str],
) -> Optional[str]:
    """Process a single table row.

    Args:
        table_idx: Table index
        row_idx: Row index
        row: Row data
        process_line: Function to process text

    Returns:
        Optional[str]: Processed row text or None
    """
    row_text = format_row(table_idx, row_idx, row)
    if not row_text:
        return None

    processed = process_line(row_text)
    return processed if processed else None


def process_tables(
    doc_data: DocumentData,
    process_line: Callable[[str], str],
) -> List[str]:
    """Process table data into text chunks.

    Args:
        doc_data: Document data
        process_line: Function to process each line

    Returns:
        List[str]: Processed table chunks
    """
    chunks = []

    for table_idx, df in enumerate(doc_data['dataframes']):
        for row_idx, row in df.iterrows():
            if processed := process_table_row(table_idx, row_idx, row, process_line):
                chunks.append(processed)

    return chunks
