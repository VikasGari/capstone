# Centralized Prompt Templates Repository

QUERY_TRANSFORM_PROMPT = """
Analyze the following user query asking about trading rules, brokerage fees, margins, or accounts.
Decompose it into a list of 1 to 3 distinct search terms or sub-queries optimized for search index retrieval (BM25 and Vector search).
Expand abbreviations like F&O (Futures & Options), MTM (Mark to Market), POA (Power of Attorney), KYC (Know Your Customer), or DP (Depository Participant) to their full terms.

User Query: "{query}"

Output the queries as a valid JSON list of strings. Do not add markdown code fences, backticks, or any conversational text.
Example Output:
["futures and options margin requirements", "span margin vs exposure margin"]
"""

def get_generation_prompt(query: str, retrieved_chunks: list[dict], refusal_msg: str) -> str:
    """Constructs the prompt string with Context Blocks and domain guardrails."""
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

    return f"""
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
