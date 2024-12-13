"""Table formatting utilities for the LLaMA RAG system."""

from typing import Callable, List, Optional

import pandas as pd

from RAG.types import DocumentData


class TableFormatter:
    """Handles table formatting operations."""

    @classmethod
    def format_cell(cls, column: str, cell_content: str) -> str:
        """Format a cell value with its column name.

        Args:
            column: Column name
            cell_content: Cell content

        Returns:
            str: Formatted cell value
        """
        return '{0}: {1}'.format(column, cell_content)

    @classmethod
    def format_table_prefix(cls, table_idx: int) -> str:
        """Format table prefix.

        Args:
            table_idx: Table index

        Returns:
            str: Formatted table prefix
        """
        return 'Table {0}'.format(table_idx + 1)

    @classmethod
    def format_row_prefix(cls, row_idx: int) -> str:
        """Format row prefix.

        Args:
            row_idx: Row index

        Returns:
            str: Formatted row prefix
        """
        return 'Row {0}'.format(row_idx + 1)

    @classmethod
    def format_location(cls, table_idx: int, row_idx: int) -> str:
        """Format table and row location.

        Args:
            table_idx: Table index
            row_idx: Row index

        Returns:
            str: Formatted location string
        """
        table_prefix = cls.format_table_prefix(table_idx)
        row_prefix = cls.format_row_prefix(row_idx)
        return '{0}, {1}:'.format(table_prefix, row_prefix)

    @classmethod
    def process_row_data(cls, row_data: pd.Series) -> List[str]:
        """Process row data into formatted cells.

        Args:
            row_data: Row data

        Returns:
            List[str]: List of formatted cells
        """
        formatted_cells = []
        for column, cell_content in row_data.items():
            if pd.notna(cell_content) and str(cell_content).strip():
                formatted_cells.append(cls.format_cell(column, str(cell_content)))
        return formatted_cells

    @classmethod
    def format_row(
        cls,
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
        formatted_cells = cls.process_row_data(row_data)
        if not formatted_cells:
            return None

        location = cls.format_location(table_idx, row_idx)
        cell_text = ' | '.join(formatted_cells)
        return '{0} {1}'.format(location, cell_text)


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
    row_text = TableFormatter.format_row(table_idx, row_idx, row)
    if not row_text:
        return None

    processed_text = process_line(row_text)
    return processed_text if processed_text else None


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
            processed_row = process_table_row(table_idx, row_idx, row, process_line)
            if processed_row:
                chunks.append(processed_row)

    return chunks
