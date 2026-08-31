import os
import json
from google import genai
from google.genai import types
from config.config_manager import ConfigManager
from src.helpers import QUERY_TRANSFORM_PROMPT

class QueryTransformer:
    """
    Transforms, expands, or decomposes multi-part and ambiguous user queries
    before retrieval, using Google Gemini.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration section directly from global config
        gen_cfg = self.config_manager.get_section("generation")
        
        # Extract configurations directly into distinct properties
        self.model_name = gen_cfg.get("primary_model")
        self.temperature = gen_cfg.get("temperature")
            
        # Initialize Google GenAI client if API key is present
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Google GenAI Client in QueryTransformer: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found in environment. QueryTransformer will fall back to original queries.")

    def transform(self, query: str) -> list[str]:
        """
        Decomposes or expands a user query.
        Returns a list of search queries. If the API client is not configured
        or the model call fails, falls back to returning the original query in a single-item list.
        """
        if not self.client:
            return [query]
            
        prompt = QUERY_TRANSFORM_PROMPT.format(query=query)
        try:
            # Generate content using structured JSON mode
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=float(self.temperature),
                    response_mime_type="application/json"
                )
            )
            text = response.text.strip()
            queries = json.loads(text)
            if isinstance(queries, list) and len(queries) > 0 and all(isinstance(q, str) for q in queries):
                print(f"Query expansion: '{query}' -> {queries}")
                return queries
        except Exception as e:
            print(f"Error during query transformation: {e}. Falling back to original query.")
            
        return [query]
