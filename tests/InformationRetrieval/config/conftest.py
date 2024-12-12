import pytest

from InformationRetrieval.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    return ConfigManager()


@pytest.fixture
def temp_config_file(tmp_path):
    config_content = """
    sentence_markers:
        en: ['.', '!', '?']
        ru: ['.', '!', '?', '...']
    """
    config_file = tmp_path / 'test_config.yaml'
    config_file.write_text(config_content)
    return config_file
