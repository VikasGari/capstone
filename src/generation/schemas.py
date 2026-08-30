from pydantic import BaseModel, Field

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
