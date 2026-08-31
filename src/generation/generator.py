from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_manager import ConfigManager
from src.generation.schemas import GroundedAnswer
from src.helpers import GENERATION_PROMPT, format_context_blocks

class GroundedGenerator:
    """
    Generates grounded answers based on retrieved context.
    Orchestrated via LangChain Expression Language (LCEL) with ChatGoogleGenerativeAI.
    Enforces Pydantic structured output with automatic model fallback chains.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration sections from global config
        gen_cfg = self.config_manager.get_section("generation")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        # Extract configurations directly into distinct properties
        self.primary_model = gen_cfg.get("primary_model")
        self.fallback_model = gen_cfg.get("fallback_model")
        self.temperature = gen_cfg.get("temperature")
        self.max_retries = gen_cfg.get("max_retries")
        self.min_relevance_score = retrieval_cfg.get("min_relevance_score")
        
        # Initialize LangChain chat models if API key is present
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")
        self.chain = None
        
        if self.api_key:
            try:
                # Primary model runnable with structured output
                primary_llm = ChatGoogleGenerativeAI(
                    model=self.primary_model,
                    temperature=float(self.temperature),
                    google_api_key=self.api_key,
                    max_retries=int(self.max_retries)
                ).with_structured_output(GroundedAnswer)
                
                # Fallback model runnable with structured output
                fallback_llm = ChatGoogleGenerativeAI(
                    model=self.fallback_model,
                    temperature=float(self.temperature),
                    google_api_key=self.api_key,
                    max_retries=int(self.max_retries)
                ).with_structured_output(GroundedAnswer)
                
                # Compose LCEL chain with automatic fallback orchestration
                structured_model = primary_llm.with_fallbacks([fallback_llm])
                self.chain = GENERATION_PROMPT | structured_model
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain ChatGoogleGenerativeAI in GroundedGenerator: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found in environment. GroundedGenerator will fail safely on all generation calls.")

    def _create_refusal_answer(self, answer_text: str) -> GroundedAnswer:
        """Helper to construct a standard GroundedAnswer refusal response."""
        return GroundedAnswer(
            answer=answer_text,
            citations=[],
            applicable_rules=[],
            thresholds_and_timelines=[],
            required_actions=[],
            grounding_confidence=0.0,
            is_sufficient=False
        )

    def generate(self, query: str, retrieved_chunks: list[dict], model_name: str = None) -> GroundedAnswer:
        """
        Generates structured grounded answers using LangChain LCEL chain.
        Fails safely on network errors or API exhaustion.
        """
        refusal_msg = "I apologize, but I cannot find information in the available trading policies to answer your query. Please refer to the official exchange site or contact support."
        
        # Guardrail 1: Empty context
        if not retrieved_chunks:
            return self._create_refusal_answer(refusal_msg)
            
        # Guardrail 2: Missing API credentials
        if not self.api_key or not self.chain:
            return self._create_refusal_answer("Error: Gemini API Client not initialized. Please verify GEMINI_API_KEY environment variable.")
            
        context_str = format_context_blocks(retrieved_chunks)
        payload = {
            "query": query,
            "context_str": context_str,
            "refusal_msg": refusal_msg
        }

        try:
            # If a specific custom model is requested (e.g. during evaluation model comparisons)
            if model_name and model_name != self.primary_model:
                custom_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=float(self.temperature),
                    google_api_key=self.api_key,
                    max_retries=int(self.max_retries)
                ).with_structured_output(GroundedAnswer)
                custom_chain = GENERATION_PROMPT | custom_llm
                grounded_ans: GroundedAnswer = custom_chain.invoke(payload)
            else:
                # Invoke the default LCEL fallback chain
                grounded_ans: GroundedAnswer = self.chain.invoke(payload)
                
            # Apply standard non-advisory disclaimer
            disclaimer = "\n\n*Disclaimer: This information is derived from brokerage and exchange policy documents for informational purposes only. It does not constitute financial, investment, or legal advice.*"
            if grounded_ans.is_sufficient and disclaimer not in grounded_ans.answer:
                grounded_ans.answer += disclaimer
                
            return grounded_ans
            
        except Exception as e:
            print(f"Error during LangChain LCEL generation: {e}. Falling back to safe response.")
            return self._create_refusal_answer(
                f"I apologize, but I am currently unable to process your request due to system connection errors: {str(e)}. Please try again shortly."
            )
