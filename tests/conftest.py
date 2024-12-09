"""Root conftest for pytest configuration."""

from tests.path_setup import setup_path


def pytest_configure():
    """Configure test environment."""
    setup_path()
