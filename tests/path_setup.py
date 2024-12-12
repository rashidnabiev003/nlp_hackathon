"""Configure Python path for tests."""

import sys
from pathlib import Path


def setup_path() -> None:
    """Add src directory to Python path."""
    # Get the absolute path to the src directory
    src_path = str(Path(__file__).parent.parent / 'src')

    # Add to Python path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
