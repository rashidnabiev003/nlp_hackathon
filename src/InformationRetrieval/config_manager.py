from pathlib import Path
from typing import Optional, Union


class ConfigManager:
    """Manages configuration settings for the IR system."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize config manager.

        Args:
            config_path: Path to config file (optional)
        """
        self.language_map = {
            'en': 'english',
            'ru': 'russian',
            # Add more languages as needed
        }

    def get_language(self, language_code: str) -> str:
        """Get LangChain language string for specified language code.

        Args:
            language_code: Language code

        Returns:
            str: Language string. Defaults to 'english' if language not found.
        """
        return self.language_map.get(language_code, 'english')
