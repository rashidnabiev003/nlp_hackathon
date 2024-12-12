"""Tests for config manager."""

from InformationRetrieval.config_manager import ConfigManager

# Valid language strings
VALID_LANGUAGES = frozenset(
    (
        'english',
        'russian',
    ),
)


def test_config_manager_initialization():
    """Test config manager initialization."""
    config = ConfigManager()
    assert isinstance(config.language_map, dict)
    assert 'en' in config.language_map
    assert 'ru' in config.language_map


def test_get_language_english():
    """Test English language mapping."""
    config = ConfigManager()
    lang = config.get_language('en')
    assert lang == 'english'


def test_get_language_russian():
    """Test Russian language mapping."""
    config = ConfigManager()
    lang = config.get_language('ru')
    assert lang == 'russian'


def test_get_language_unknown():
    """Test fallback for unknown language codes."""
    config = ConfigManager()
    lang = config.get_language('unknown')
    assert lang == 'english'


def test_language_map_consistency():
    """Test language mapping consistency."""
    config = ConfigManager()
    for _, lang_str in config.language_map.items():
        assert isinstance(lang_str, str)
        assert lang_str in VALID_LANGUAGES
