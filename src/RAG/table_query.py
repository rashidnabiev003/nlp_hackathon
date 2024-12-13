"""Table querying functionality for the LLaMA RAG system."""

import logging
import re
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_table_cell(
    dataframes: List[pd.DataFrame],
    table_index: int,
    row_index: int,
    column_name: str,
) -> str:
    """Get cell value from specified table location.

    Args:
        dataframes: List of dataframes
        table_index: 1-based table index
        row_index: 1-based row index
        column_name: Column name

    Returns:
        str: Cell value or error message
    """
    try:
        if 0 < table_index <= len(dataframes):
            df = dataframes[table_index - 1]
            if 0 < row_index <= len(df) and column_name in df.columns:
                cell_value = df.iloc[row_index - 1][column_name]
                if pd.notna(cell_value) and str(cell_value).strip():
                    return str(cell_value)
    except Exception as error:
        logger.error('Error accessing table cell: %s', error)

    return 'Недостаточно информации.'


def build_cell_pattern() -> str:
    """Build regex pattern for cell requests.

    Returns:
        str: Compiled regex pattern
    """
    table_pattern = r'таблиц[а|е]\s*(\d+)'
    row_pattern = r'строк[а|е]\s*(\d+)'
    column_pattern = r"столбец\s*'([^']+)'|столбец\s*\"([^\"]+)\""

    return f'{table_pattern}.*{row_pattern}.*{column_pattern}'


def parse_cell_request(query: str) -> Optional[Tuple[int, int, str]]:
    """Parse direct cell request from query.

    Args:
        query: User query string

    Returns:
        Optional[Tuple[int, int, str]]: (table index, row index, column name) if found
    """
    pattern = build_cell_pattern()
    match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    table_num = int(match.group(1))
    row_num = int(match.group(2))
    col_name = match.group(3) if match.group(3) else match.group(4)
    return table_num, row_num, col_name.strip()
