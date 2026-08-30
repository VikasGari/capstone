import json
import time
from google import genai
from google.genai import types
from config.config_manager import ConfigManager
from src.generation.schemas import GroundedAnswer, Citation
from src.generation.prompts import get_generation_prompt

class GroundedGenerator:
    """
    Generates grounded answers based on retrieved context.
    Enforces Pydantic structured output utilizing Gemini.
    Implements retries and safe fallback responses.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration sections from global config
        gen_cfg = self.config_manager.get_section("generation")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        # Extract configurations directly into distinct properties (no self.config dict lookup)
        self.primary_model = local_overrides.get("primary_model") if local_overrides and "primary_model" in local_overrides else gen_cfg.get("primary_model")
        self.fallback_model = local_overrides.get("fallback_model") if local_overrides and "fallback_model" in local_overrides else gen_cfg.get("fallback_model")
        self.temperature = local_overrides.get("temperature") if local_overrides and "temperature" in local_overrides else gen_cfg.get("temperature")
        self.max_retries = local_overrides.get("max_retries") if local_overrides and "max_retries" in local_overrides else gen_cfg.get("max_retries")
        self.min_relevance_score = local_overrides.get("min_relevance_score") if local_overrides and "min_relevance_score" in local_overrides else retrieval_cfg.get("min_relevance_score")
        
        # Initialize Google GenAI client if API key is present
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Google GenAI Client in GroundedGenerator: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found in environment. GroundedGenerator will fail safely on all generation calls.")

    def generate(self, query: str, retrieved_chunks: list[dict], model_name: str = None) -> GroundedAnswer:
        """
        Generates structured grounded answers, citations, and rules.
        Fails safely on network errors or API exhaustion.
        """
        # Refusal fallback message
        refusal_msg = "I apologize, but I cannot find information in the available trading policies to answer your query. Please refer to the official exchange site or contact support."
        
        # Guardrail: Check if we have any retrieved context
        if not retrieved_chunks:
            return GroundedAnswer(
                answer=refusal_msg,
                citations=[],
                applicable_rules=[],
                thresholds_and_timelines=[],
                required_actions=[],
                grounding_confidence=0.0,
                is_sufficient=False
            )
            
        prompt = get_generation_prompt(query, retrieved_chunks, refusal_msg)

        # Decide which model to use
        if model_name is None:
            model_name = self.primary_model
            
        max_retries = int(self.max_retries)
        temperature = float(self.temperature)
        
        # Safe offline fallback if no client exists
        if not self.client:
            return GroundedAnswer(
                answer="Error: Gemini API Client not initialized. Please verify GEMINI_API_KEY environment variable.",
                citations=[],
                applicable_rules=[],
                thresholds_and_timelines=[],
                required_actions=[],
                grounding_confidence=0.0,
                is_sufficient=False
            )

        # Retry loop for resilience (NFR-05)
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GroundedAnswer,
                        temperature=temperature
                    )
                )
                
                text = response.text.strip()
                ans_dict = json.loads(text)
                
                # Enforce Pydantic validation on parsed dictionary
                grounded_ans = GroundedAnswer(**ans_dict)
                
                # Post-processing: append standard disclaimer to answer text
                disclaimer = "\n\n*Disclaimer: This information is derived from brokerage and exchange policy documents for informational purposes only. It does not constitute financial, investment, or legal advice.*"
                if grounded_ans.is_sufficient and disclaimer not in grounded_ans.answer:
                    grounded_ans.answer += disclaimer
                    
                return grounded_ans
                
            except Exception as e:
                last_error = e
                print(f"API Attempt {attempt+1} failed with error: {e}. Retrying...")
                time.sleep(1.5 * (attempt + 1)) # Backoff

        # Graceful Fallback on Repeated Failures
        print(f"All {max_retries} attempts failed. Falling back to safe response.")
        
        # If primary failed, we try a single attempt with fallback model
        fallback_model = self.fallback_model
        if model_name != fallback_model:
            try:
                print(f"Attempting fallback model: {fallback_model}")
                response = self.client.models.generate_content(
                    model=fallback_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GroundedAnswer,
                        temperature=temperature
                    )
                )
                ans_dict = json.loads(response.text.strip())
                grounded_ans = GroundedAnswer(**ans_dict)
                disclaimer = "\n\n*Disclaimer: This information is derived from brokerage and exchange policy documents for informational purposes only. It does not constitute financial, investment, or legal advice.*"
                if grounded_ans.is_sufficient and disclaimer not in grounded_ans.answer:
                    grounded_ans.answer += disclaimer
                return grounded_ans
            except Exception as fallback_err:
                print(f"Fallback model also failed: {fallback_err}")
                
        # Return generic safe error object rather than crashing
        return GroundedAnswer(
            answer=f"I apologize, but I am currently unable to process your request due to system connection errors: {str(last_error)}. Please try again shortly.",
            citations=[],
            applicable_rules=[],
            thresholds_and_timelines=[],
            required_actions=[],
            grounding_confidence=0.0,
            is_sufficient=False
        )
