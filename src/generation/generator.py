import json
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config.config_manager import ConfigManager

class Citation(BaseModel):
    source: str = Field(description="The source filename or document ID of the cited rule/policy.")
    clause_id: str = Field(description="The specific clause or section ID, e.g., Clause 1.1 or Section 2.3.")
    clause_title: str = Field(description="The title of the clause/section.")
    snippet: str = Field(description="The exact short quote from the context that justifies this statement.")

class GroundedAnswer(BaseModel):
    answer: str = Field(description="The natural language answer to the query, strictly grounded in the provided context. If the context is insufficient, must be the refusal message.")
    citations: list[Citation] = Field(default=[], description="List of clause-level citations matching the answer statements. Leave empty if context is insufficient.")
    applicable_rules: list[str] = Field(default=[], description="List of rule/policy clause headings or text summaries that apply.")
    thresholds_and_timelines: list[str] = Field(default=[], description="List of specific numerical limits, percentages, cutoffs, or settlement timelines mentioned in the answer.")
    required_actions: list[str] = Field(default=[], description="List of actionable items required by the user or brokerage based on the rule (e.g., 'Submit Re-KYC', 'Deposit funds before 23:59').")
    grounding_confidence: float = Field(description="Confidence rating of grounding (0.0 to 1.0) based on how well the context covers the query.")
    is_sufficient: bool = Field(description="True if context has enough info to answer. False if insufficient context and system has abstained.")

class GroundedGenerator:
    """
    Generates grounded answers based on retrieved context.
    Enforces Pydantic structured output utilizing Gemini.
    Implements retries and safe fallback responses.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Local defaults
        local_defaults = {
            "primary_model": "gemini-1.5-flash",
            "fallback_model": "gemini-1.5-pro",
            "temperature": 0.0,
            "max_retries": 3,
            "min_relevance_score": 0.1
        }
        
        # Merge with global overrides
        gen_cfg = self.config_manager.get_section("generation")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        self.config = local_defaults.copy()
        
        if "primary_model" in gen_cfg:
            self.config["primary_model"] = gen_cfg["primary_model"]
        if "fallback_model" in gen_cfg:
            self.config["fallback_model"] = gen_cfg["fallback_model"]
        if "temperature" in gen_cfg:
            self.config["temperature"] = gen_cfg["temperature"]
        if "max_retries" in gen_cfg:
            self.config["max_retries"] = gen_cfg["max_retries"]
        if "min_relevance_score" in retrieval_cfg:
            self.config["min_relevance_score"] = retrieval_cfg["min_relevance_score"]
            
        if local_overrides:
            self.config.update(local_overrides)
            
        # Initialize Google GenAI client
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Google GenAI Client in GroundedGenerator: {e}")
        else:
            print("Warning: GEMINI_API_KEY not found in environment. Generator will run in offline fallback mode.")

    def generate(self, query: str, retrieved_chunks: list[dict], model_name: str = None) -> GroundedAnswer:
        """
        Generates structured grounded response from query and retrieved context.
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
            
        # Format the context block for the model
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            clause_id = meta.get("clause_id", "General")
            clause_title = meta.get("clause_title", "General")
            
            context_str += f"--- CONTEXT BLOCK {idx+1} ---\n"
            context_str += f"Source Doc: {source}\n"
            context_str += f"Clause/Section ID: {clause_id}\n"
            context_str += f"Clause/Section Title: {clause_title}\n"
            context_str += f"Content:\n{chunk['document']}\n\n"

        prompt = f"""
You are an expert Brokerage Rules & Trading Policy Assistant. Your task is to answer the user query based ONLY on the provided Context Blocks.

Domain Guardrails:
1. Base your answer strictly on the facts present in the Context Blocks. Do not assume, extrapolate, or bring in outside knowledge.
2. If the Context Blocks do not contain sufficient information to answer the query, set `is_sufficient = false` and set the `answer` field to: "{refusal_msg}"
3. Do not offer definitive legal, financial, or investment advice. Include a passive notice/disclaimer at the end of the answer that this is informational only.
4. Each statement in your answer must be supported by a citation mapping back to the specific Context Block. In the `citations` list, include the exact snippet/quote of text that justifies your statements, along with the source file, clause ID, and clause title.

Provided Context Blocks:
{context_str}

User Query: "{query}"
"""

        # Decide which model to use
        if model_name is None:
            model_name = self.config["primary_model"]
            
        max_retries = int(self.config["max_retries"])
        temperature = float(self.config["temperature"])
        
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
                
                # Parse output
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
        
        # If primary failed, we try a single attempt with fallback model (e.g., gemini-1.5-pro)
        fallback_model = self.config["fallback_model"]
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
