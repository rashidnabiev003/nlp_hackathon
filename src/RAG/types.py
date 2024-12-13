"""Type definitions for the LLaMA RAG system."""

from typing import List, Tuple, TypedDict

import pandas as pd

# Type aliases for table structures
TableRow = List[str]
TableData = List[TableRow]
RawTableList = List[TableData]


class TableList(List[List[List[str]]]):
    """Type for table data structure."""


TableExtractionResult = Tuple[RawTableList, List[pd.DataFrame]]


class DocumentData(TypedDict):
    """Structure for document data."""

    paragraphs: List[str]
    tables: TableList  # List of tables, each table is a list of rows
    dataframes: List[pd.DataFrame]
