import os
import json
from google import genai
from google.genai import types
from config.config_manager import ConfigManager

class QueryTransformer:
    """
    Transforms, expands, or decomposes multi-part and ambiguous user queries
    before retrieval, using Google Gemini.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration section directly from global config
        gen_cfg = self.config_manager.get_section("generation")
        
        # Extract configurations directly into distinct properties (no self.config dict lookup)
        self.model_name = local_overrides.get("model_name") if local_overrides and "model_name" in local_overrides else gen_cfg.get("primary_model")
        self.temperature = local_overrides.get("temperature") if local_overrides and "temperature" in local_overrides else gen_cfg.get("temperature")
            
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
            
        prompt = f"""
Analyze the following user query asking about trading rules, brokerage fees, margins, or accounts.
Decompose it into a list of 1 to 3 distinct search terms or sub-queries optimized for search index retrieval (BM25 and Vector search).
Expand abbreviations like F&O (Futures & Options), MTM (Mark to Market), POA (Power of Attorney), KYC (Know Your Customer), or DP (Depository Participant) to their full terms.

User Query: "{query}"

Output the queries as a valid JSON list of strings. Do not add markdown code fences, backticks, or any conversational text.
Example Output:
["futures and options margin requirements", "span margin vs exposure margin"]
"""
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
