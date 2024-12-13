"""Type definitions for the LLaMA RAG system."""

from typing import List, TypedDict

import pandas as pd


class TableList(List[List[List[str]]]):
    """Type for table data structure."""


class DocumentData(TypedDict):
    """Structure for document data."""

    paragraphs: List[str]
    tables: TableList  # List of tables, each table is a list of rows
    dataframes: List[pd.DataFrame]
