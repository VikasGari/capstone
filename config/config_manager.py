import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """
    Manages configuration for the RAG system.
    Loads global configurations from global_config.yaml and provides them to components.
    Ensures that global configuration overrides local default configurations.
    Also handles environment variables (via python-dotenv).
    """
    def __init__(self, global_config_path: str = None):
        # Load environment variables from .env
        load_dotenv()
        
        # Determine the global configuration path
        if global_config_path is None:
            # Default location: config/global_config.yaml relative to workspace root
            project_root = Path(__file__).resolve().parent.parent
            global_config_path = project_root / "config" / "global_config.yaml"
        
        self.global_config_path = Path(global_config_path)
        self.global_config = self._load_yaml_config()

    def _load_yaml_config(self) -> dict:
        """Loads the YAML global config file."""
        if not self.global_config_path.exists():
            print(f"Warning: Global configuration file not found at {self.global_config_path}. Using empty defaults.")
            return {}
        
        try:
            with open(self.global_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except Exception as e:
            print(f"Error loading global config: {e}. Using empty defaults.")
            return {}

    def get_section(self, section_name: str, local_defaults: dict = None) -> dict:
        """
        Retrieves a configuration section from the global YAML configuration.
        Optionally merges local defaults, and overrides with environment variables.
        """
        global_section = self.global_config.get(section_name, {})
        merged_config = {}
        if local_defaults:
            merged_config.update(local_defaults)
            
        if isinstance(global_section, dict):
            for key, val in global_section.items():
                merged_config[key] = val
        
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
