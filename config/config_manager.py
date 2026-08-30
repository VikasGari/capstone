import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """
    Manages configuration for the RAG system.
    Loads configurations from config.yaml and provides them to components.
    """
    def __init__(self, config_path: str = None):
        load_dotenv()
        
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config()

    def _load_yaml_config(self) -> dict:
        """Loads the YAML config file."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                return config_data if config_data else {}
        except Exception:
            return {}

    def get_section(self, section_name: str) -> dict:
        """Retrieves a configuration section from the YAML configuration."""
        section = self.config.get(section_name, {})
        return dict(section) if isinstance(section, dict) else {}

    def get_env_var(self, name: str, default: str = None) -> str:
        """Helper to get environment variables directly (like GEMINI_API_KEY)."""
        return os.environ.get(name, default)
