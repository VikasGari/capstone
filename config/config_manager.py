import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """
    Manages configuration for the RAG system.
    Loads configurations from config.yaml and provides them to components.
    Also handles environment variables (via python-dotenv).
    """
    def __init__(self, config_path: str = None):
        # Load environment variables from .env
        load_dotenv()
        
        # Determine the configuration path
        if config_path is None:
            # Default location: config/config.yaml relative to workspace root
            project_root = Path(__file__).resolve().parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config()

    def _load_yaml_config(self) -> dict:
        """Loads the YAML config file."""
        if not self.config_path.exists():
            print(f"Warning: Configuration file not found at {self.config_path}. Using empty defaults.")
            return {}
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                return config_data if config_data else {}
        except Exception as e:
            print(f"Error loading config: {e}. Using empty defaults.")
            return {}

    def get_section(self, section_name: str) -> dict:
        """
        Retrieves a configuration section from the YAML configuration.
        Keys can be overridden by environment variables (e.g. EMBEDDING_MODEL_NAME).
        """
        section = self.config.get(section_name, {})
        merged_config = dict(section) if isinstance(section, dict) else {}
        
        # Override specific keys with environment variables if available
        # e.g., GEMINI_API_KEY
        for key in list(merged_config.keys()):
            env_key = f"{section_name.upper()}_{key.upper()}"
            if env_key in os.environ:
                merged_config[key] = os.environ[env_key]
        
        return merged_config

    def get_env_var(self, name: str, default: str = None) -> str:
        """Helper to get environment variables directly."""
        return os.environ.get(name, default)
