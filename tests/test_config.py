import os
from config.config_manager import ConfigManager

def test_config_overrides():
    # Test normal configuration loading
    cm = ConfigManager()
    
    # Check default section merging
    embedding_config = cm.get_section("embedding")
    assert "model_name" in embedding_config
    assert embedding_config["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
