# Acceptance Criteria Spec & Testability Matrix

This specification maps the Functional Acceptance Criteria (AC) and Non-Functional Requirements (NFR) to their corresponding implementation modules and verification test cases.

---

## 1. Functional Acceptance Criteria (AC)

| ID | Description | Mapped Verification/Test Location |
|---|---|---|
| **AC-01** | Ingests a synthetic corpus of $\ge 30$ documents across 5 segments (exchange rulebook, margin policy, settlement procedures, trading terms, fees & brokerage schedule) into a persisted vector index; ingestion is re-runnable and idempotent. | Verified by `tests/test_ingestion.py` which checks that run completes without duplicate indexing, and is idempotent. |
| **AC-02** | A user can ask a natural-language query and receive an answer grounded only in the corpus, with $\ge 1$ clause-level citation (document + section/clause id) per answer. | Verified by `tests/test_generation.py` checking the presence of citations in the structured outputs, and evaluation tests. |
| **AC-03** | Retrieval combines lexical (BM25) and semantic search and fuses the two result sets (e.g., Reciprocal Rank Fusion) before generation. | Verified by `tests/test_retrieval.py` testing the hybrid retriever output ranks and fusion logic. |
| **AC-04** | Retrieved candidates are reranked (cross-encoder) before the top-K is passed to the generator. | Verified by `tests/test_retrieval.py` verifying that cross-encoder scores adjust candidate ranking order. |
| **AC-05** | When the corpus does not support an answer, the system abstains or flags low confidence rather than fabricating. | Evaluated in `data/golden_set.json` (unanswerable/out-of-corpus queries) and checked in `tests/test_generation.py`. |
| **AC-06** | Answers are returned as a validated structured object (Pydantic / JSON schema) containing answer text, citations, applicable rule / policy clause, any threshold or timeline, and the required action, and a grounding / confidence indicator. | Verified by `tests/test_generation.py` validating the JSON outputs against `GroundedAnswer` Pydantic models. |
| **AC-07** | Multi-part or ambiguous queries are transformed (rewrite / expansion / decomposition) before retrieval. | Verified by querying the `QueryTransformer` class instance with multi-part test cases in `tests/test_retrieval.py`. |
| **AC-08** | A golden evaluation set of $\ge 20$ questions with reference answers / expected contexts is committed with a re-runnable scoring script. | Golden set committed under `data/golden_set.json`. Run script is `src/evaluation/harness.py`. |
| **AC-09** | RAGAS metrics (context precision, context recall, faithfulness, answer relevancy) are computed and the numeric results committed as a report. | Evaluated by `src/evaluation/harness.py` and output committed to `docs/eval_report.md`. |
| **AC-10** | At least two candidate LLMs are evaluated on the custom eval set and a comparison (metrics + selection rationale) is committed. | Evaluated by `src/evaluation/harness.py` comparing Gemini 1.5 Flash vs Gemini 1.5 Pro and committed to `docs/model_comparison.md`. |

---

## 2. Non-Functional Requirements (NFR)

| ID | Description | Mapped Verification/Test Location |
|---|---|---|
| **NFR-01** | No secrets or API keys committed; configuration via environment variables with a committed `.env.example`. | Checked by scanning project for `.env` and hardcoded API keys. Template committed as `.env.example`. |
| **NFR-02** | Pipeline and evaluation run end-to-end from a single documented command with committed sample data and a README quick-start. | Run via `python main.py --all` or individually documented commands in `README.md`. |
| **NFR-03** | All data is synthetic; any PII is synthetic, masked where shown, and never written to logs in plaintext. | Handled via custom generation scripts and masked logging. |
| **NFR-04** | Retrieval parameters (chunk size / overlap, top-K, thresholds, reranker settings) are externalized in config, not hard-coded inline. | Loaded from `config/config.yaml` and processed via `ConfigManager`. |
| **NFR-05** | Provider / model calls implement basic retries and fail gracefully — a clear error or safe fallback response on repeated failure — rather than crashing. | Wrapped with tenacity retries or custom exception handlers in `GroundedGenerator`. |
| **NFR-06** | Every quality claim is reproducible — the eval script and dataset are committed so metrics can be re-derived. | Python scripts under `src/evaluation/` execute evaluation locally on committed `data/golden_set.json`. |
| **NFR-07** | Cost and latency are addressed at concept level only. | Briefly documented in `docs/walkthrough.md`. |
| **NFR-08** | Basic logging of queries and answers is in place; no answer is returned without provenance above the abstention threshold. | Handled in `main.py` and the FastAPI server, logging to `app.log`. |
