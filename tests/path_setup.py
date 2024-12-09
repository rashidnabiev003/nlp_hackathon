"""Configure Python path for tests."""

import sys
from pathlib import Path


def setup_path() -> None:
    """Add src to Python path for test discovery."""
    src_path = str(Path(__file__).parent.parent / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
