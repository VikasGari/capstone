# Business Case: Brokerage Rules & Trading Policy Assistant

* **Business Case Title:** Brokerage Rules & Trading Policy Assistant — Retrieval-Augmented Assistant
* **Business Case ID:** AAIE_007_CAP
* **Domain:** Stock Markets — Brokerage & Trading Operations
* **Track:** Gen AI Core + RAG Engineering

---

## 1. Problem Statement
In brokerage operations, support staff and customers frequently inquire about trading rules, margin requirements for derivatives (Futures & Options), settlement cycles, account opening terms, and fee schedules. Providing inaccurate information can lead to financial losses, compliance violations, and regulatory penalties. 

Currently, staff must search through disjointed exchange circulars, internal margin policies, and term sheets manually, which is slow and error-prone. This project delivers an intelligent, Retrieval-Augmented Generation (RAG) assistant that provides instant, grounded answers with clause-level citations to source documents.

---

## 2. Target Users
1. **Customer Support Executives:** Need rapid, precise answers to customer queries regarding margin requirements, trade settlements, and payout timelines.
2. **Operations & Compliance Officers:** Need to verify policy alignment against exchange rulebooks.
3. **Active Traders / Clients (Conceptual):** Seek self-service clarification on brokerage fees, trading hours, and account terms.

---

## 3. Corpus Description
The assistant operates over a synthetic corpus containing at least 30 documents across five categories:
1. **Exchange Rulebook (Analogue):** Trading hours, order types, circuit breakers, contract specifications, and block trade rules.
2. **Brokerage Margin & F&O Policy:** SPAN and Exposure margins, Peak Margin requirements, Mark-to-Market (MTM) settlement, margin calls, and auto-square-off conditions.
3. **Settlement and Payout Procedures:** T+1 equity settlement, funds/securities pay-in and pay-out timelines, bank integration cutoffs, and auction settlement rules.
4. **Account-Opening & Trading Terms:** KYC documents, inactive account policy, Power of Attorney (POA) limitations, and account closure procedures.
5. **Fees & Brokerage Schedule:** Flat/percentage brokerage per segment, Securities Transaction Tax (STT), Goods and Services Tax (GST), exchange transaction charges, and SEBI turnover fees.

---

## 4. Domain Guardrails & Guidelines
* **Non-Advisory Guardrail:** The system must include a disclaimer that the information provided is for educational/informational purposes only and does not constitute financial or legal advice.
* **Strict Grounding:** The assistant must base its responses solely on the ingested corpus.
* **Abstention Policy:** If a user query falls outside the corpus or is ambiguous, the assistant must refuse to answer or state its inability to answer, rather than hallucinate.
* **Traceability:** Every response must be accompanied by explicit, clause-level citations pointing to the source document and section/clause ID.

---

## 5. Success Metrics
The system's performance will be quantified using RAGAS metrics over a golden evaluation set of 20+ scenarios:
* **Context Recall:** Ability of the retrieval system to fetch all relevant contexts. Target: $\ge 0.85$.
* **Context Precision:** Relevance of the retrieved chunks to the user query. Target: $\ge 0.85$.
* **Faithfulness:** Groundedness of the generated answer in the retrieved context (no hallucinations). Target: $\ge 0.90$.
* **Answer Relevancy:** How well the generated answer addresses the user's question. Target: $\ge 0.90$.
