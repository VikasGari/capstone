import os
from unittest import mock
from config.config_manager import ConfigManager

def test_config_overrides():
    # Test normal configuration loading
    cm = ConfigManager()
    
    # Check default section merging
    embedding_config = cm.get_section("embedding")
    assert "model_name" in embedding_config
    assert embedding_config["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_environment_variable_override():
    # Set mock environment variable
    with mock.patch.dict(os.environ, {"EMBEDDING_MODEL_NAME": "env-override-model"}):
        cm = ConfigManager()
        embedding_config = cm.get_section("embedding")
        assert embedding_config["model_name"] == "env-override-model"
