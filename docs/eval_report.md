# RAGAS Evaluation Report

This report documents the performance metrics and retrieval quality evaluations for the Brokerage Rules & Trading Policy Assistant, evaluated over a golden set of 21 scenarios.

## 1. RAGAS Performance Summary (Judge: Google Gemini)

| Metric | Target | Gemini 3.5 Flash (Primary) | Gemini 2.5 Pro (Fallback) |
|---|---|---|---|
| **Faithfulness** | $\ge 0.90$ | 0.094 | 0.018 |
| **Answer Relevancy** | $\ge 0.90$ | 0.210 | 0.170 |
| **Context Recall** | $\ge 0.85$ | 0.179 | 0.179 |
| **Context Precision** | $\ge 0.85$ | 1.000 | 1.000 |

---

## 2. Failure Taxonomy Analysis

### Gemini 3.5 Flash (Primary)
* **Successful Runs:** 10 / 21 (47.6%)
* **Retrieval Failures:** 11 (Missed retrieving critical policy document chunks)
* **Grounding Failures:** 0 (Hallucinations or unsupported facts generated)
* **Synthesis Failures:** 0 (Incomplete extraction of timelines, rules, or thresholds)

### Gemini 2.5 Pro (Fallback)
* **Successful Runs:** 1 / 21 (4.8%)
* **Retrieval Failures:** 20 (Missed retrieving critical policy document chunks)
* **Grounding Failures:** 0 (Hallucinations or unsupported facts generated)
* **Synthesis Failures:** 0 (Incomplete extraction of timelines, rules, or thresholds)
