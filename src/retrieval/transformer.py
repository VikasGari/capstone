from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config.config_manager import ConfigManager
from src.helpers import QUERY_TRANSFORM_PROMPT

class QueryTransformer:
    """
    Transforms, expands, or decomposes multi-part and ambiguous user queries
    before retrieval, orchestrated via a LangChain LCEL pipeline.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration section directly from global config
        gen_cfg = self.config_manager.get_section("generation")
        
        # Extract configurations directly into distinct properties
        self.model_name = gen_cfg.get("primary_model")
        self.temperature = gen_cfg.get("temperature")
            
        # Initialize LangChain model and LCEL chain if API key is present
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")
        self.chain = None
        
        if self.api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=float(self.temperature),
                    google_api_key=self.api_key
                )
                # LCEL pipeline: Prompt -> Chat Model -> JSON Output Parser
                self.chain = QUERY_TRANSFORM_PROMPT | llm | JsonOutputParser()
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain ChatGoogleGenerativeAI in QueryTransformer: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found in environment. QueryTransformer will fall back to original queries.")

    def transform(self, query: str) -> list[str]:
        """
        Decomposes or expands a user query via LangChain LCEL chain.
        Returns a list of search queries.
        """
        if not self.chain:
            return [query]
            
        try:
            queries = self.chain.invoke({"query": query})
            if isinstance(queries, list) and len(queries) > 0 and all(isinstance(q, str) for q in queries):
                print(f"Query expansion: '{query}' -> {queries}")
                return queries
        except Exception as e:
            print(f"Error during LangChain query transformation: {e}. Falling back to original query.")
            
        return [query]
