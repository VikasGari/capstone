# RAGAS Evaluation Report

This report documents the performance metrics and retrieval quality evaluations for the Brokerage Rules & Trading Policy Assistant, evaluated over a golden set of 21 scenarios.

## 1. RAGAS Performance Summary (Judge: Google Gemini)

| Metric | Target | Gemini 3.5 Flash (Primary) | Gemini 3.1 Pro (Fallback) |
|---|---|---|---|
| **Faithfulness** | $\ge 0.90$ | 0.025 | 0.027 |
| **Answer Relevancy** | $\ge 0.90$ | 0.234 | 0.257 |
| **Context Recall** | $\ge 0.85$ | 0.179 | 0.179 |
| **Context Precision** | $\ge 0.85$ | 1.000 | 1.000 |

---

## 2. Failure Taxonomy Analysis

### Gemini 3.5 Flash (Primary)
* **Successful Runs:** 14 / 21 (66.7%)
* **Retrieval Failures:** 1 (Missed retrieving critical policy document chunks)
* **Grounding Failures:** 0 (Hallucinations or unsupported facts generated)
* **Synthesis Failures:** 6 (Incomplete extraction of timelines, rules, or thresholds)

### Gemini 3.1 Pro (Fallback)
* **Successful Runs:** 15 / 21 (71.4%)
* **Retrieval Failures:** 0 (Missed retrieving critical policy document chunks)
* **Grounding Failures:** 0 (Hallucinations or unsupported facts generated)
* **Synthesis Failures:** 6 (Incomplete extraction of timelines, rules, or thresholds)
