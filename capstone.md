**# \*\*Agentic AI Engineer Pathway — Gen AI & RAG Capstone Brokerage Rules & Trading Policy Assistant — Retrieval-Augmented\*\*** 

**# \*\*Assistant\*\*** 

\_Business Case AAIE\_007\_CAP  ·  Domain: Stock Markets — Brokerage & Trading Operations  ·  Track: Gen AI Core + RAG Engineering\_ 

**## \*\*1. Project Identity\*\*** 

|**\*\*Business Case Title\*\***|Brokerage Rules & Trading Policy Assistant — Retrieval-Augmented\<br>Assistant|

\|---|---|

|**\*\*Business Case ID\*\***|AAIE\_007\_CAP\<br>i|

|**\*\*Domain\*\***|Stock Markets — Brokerage & TradingOperations|

|**\*\*Project Type\*\***|Gen AI Capstone — Gen AI Core + RAG Engineering\<br>ii|

|**\*\*Cohort\*\***|(filled byOperations team)|





**## \*\*2. Engagement Overview\*\*** 

|**\*\*Duration\*\***|15 hours|

\|---|---|

|**\*\*Format\*\***|Individual|

|**\*\*Evaluation Mode\*\***|Automated static review of the submitted Git repository against the\<br>RAG Engineering Rubric (21 parameters / 100 marks across 6\<br>categories). Rubric parameters scored with Google Gemini (not\<br>Claude); presence and numeric-threshold parameters scored\<br>deterministicallyin Python.|

|**\*\*Submission\*\***|i\<br>Push final code to your assigned Virtusa GitLab repository by the\<br>cohort cut-off.|

|**\*\*Review Output\*\***|Per-learner Excel report (Summary, Categories, Scorecard, Detailed,\<br>Improvement).|

|**\*\*Grade Bands\*\***|Distinction ≥ 90  ·  Merit 75–89  ·  Pass 60–74  ·  Not Yet Passed < 60|





**\*\*What is evaluated:\*\*** retrieval-pipeline engineering (chunking, hybrid search, fusion, reranking, query transformation), grounded generation with clause-level citations, structured output, measured retrieval quality (RAGAS on a committed golden set), model and embedding selection, and engineering reproducibility. 

**\*\*What is not evaluated:\*\*** the visual polish of any interface, model fine-tuning, or which library you pick within the approved open-source set. You are judged on retrieval quality and evidence. 

**## \*\*3. Problem Statement & Expected Solution\*\*** 

**### \*\*3.1 Problem\*\*** 

A brokerage's support and operations staff need accurate answers about trading rules, margin and derivatives (F&O) policies, and account terms — grounded strictly in the (synthetic) exchange rulebooks, the brokerage's margin policy, and account / trading terms. Questions such as “what is the margin requirement for an indexfutures position?” or “what are the settlement timelines for equity delivery?” must cite the exact rule, and the assistant must abstain when the corpus does not support an answer. 

**### \*\*3.2 Your Role\*\*** 

RAG Engineer. You design the ingestion and indexing strategy, build hybrid retrieval with reranking, engineer grounded generation with citations and domain-specific extraction (applicable rule / policy clause, any threshold or timeline, and the required action), and — critically — you measure retrieval quality with a repeatable evaluation harness and defend your model and embedding choices with data. 

**### \*\*3.3 Expected Solution\*\*** 

A working RAG application delivered as a Git repository that demonstrates: 

\- An ingestion pipeline that chunks, embeds, and indexes a synthetic brokerage-rules and trading-policy corpus into a persisted vector store, re-runnable and idempotent. 

\- A retrieval layer combining lexical (BM25) and semantic search with fusion (e.g., Reciprocal Rank Fusion) and a reranking stage before generation. 

\- A generation layer that returns validated structured output — answer, clause-level citations, applicable rule / policy clause, any threshold or timeline, and the required action, and a confidence/grounding indicator — and abstains when the corpus is insufficient. 

\- An evaluation harness that computes RAGAS metrics over a committed golden set, plus a documented comparison of at least two candidate LLMs. 

\- An evidence trail — committed eval reports, sample answers, and a reproducible eval script — proving the quality claims. 

**### \*\*3.4 Applicable Rules\*\*** 

\- **\*\*Synthetic-Data Rule.\*\*** Use only synthetic / dummy data you generate yourself — no real client, account, or trade data, and no Virtusa confidential data. 

\- **\*\*Evidence-in-Repo Rule.\*\*** Only artifacts committed to the repository are scored. Eval metrics, golden set, sample outputs, and the eval script must be committed — uncommitted results do not count. 

\- **\*\*Reproducibility Rule.\*\*** The pipeline and the evaluation must run from a single documented command with committed sample data and a README quick-start. 

\- **\*\*AC-Traceability Rule.\*\*** Each Acceptance Criterion must be referenced by at least one test or eval-set entry carrying its AC-NN identifier. 

\- **\*\*Grounding & Advice Rule.\*\*** The assistant surfaces the applicable rules/clauses with citations; it must not issue a definitive decision as legal or financial advice, and must prefer abstention over hallucination. 

\- **\*\*Open-Source & No-Docker Rule.\*\*** Use only the approved open-source stack with Google Gemini as the LLM provider. The project must build, run, and be evaluated with pip + Python alone — no Docker and no external database service. 

**## \*\*4. Technology & Framework Stack\*\*** 

The stack is fixed to an open-source toolchain with Google Gemini as the only model provider. The project must build, run, and be evaluated with pip + Python alone — no Docker or external database service. 

|**\*\*Layer\*\***\<br>i|**\*\*Approved tool(open source unless noted)\*\***|

\|---|---|

|Language/Orchestration|Python 3.11+ · LangChain/LCEL(MIT)|

|LLM Provider|Google Gemini(API)— the onlyapproved modelprovider;not Claude|

|Embeddings|Sentence-Transformers — BGE/E5/MiniLM(local,open source)|

|Vector Store|Chroma or FAISS only (open source,in-process — no Docker)|

|Lexical/Hybrid|rank\_bm25(BM25)fused with vector search|

|Reranker\<br>i|Sentence-Transformers CrossEncoder — BGE-reranker/MiniLM\<br>i|

|Evaluation|RAGAS(judge LLM = Gemini);non-LLM metrics optional|





**\*\*\<mark>Layer Approved tool (open source unless noted)\</mark>\*\*** \<mark>Interface (opt\</mark> i \<mark>onal) CLI · Streamlit · Gradio · FastAPI\</mark> 

**## \*\*5. Acceptance Criteria & Non-Functional Requirements\*\*** 

**### \*\*5.1 Functional Acceptance Criteria\*\*** 

*\_Note. Each AC must have at least one test or eval-set entry referencing its AC-NN identifier.\_* 

|**\*\*ID\*\***|**\*\*Criterion\*\***\<br>i       i|

\|---|---|

|AC-01|Ingests a synthetic corpus of ≥ 30 documents (a synthetic exchange rulebook analogue, the\<br>brokerage's margin & F&O policy, settlement and payout procedures, account-opening / trading\<br>terms, and a fees & brokerage schedule) into a persisted vector index; ingestion is re-runnable and\<br>idempotent.\<br>i|

|AC-02|A user can ask a natural-language question and receive an answer grounded only in the corpus, with\<br>≥ 1 clause-level citation(document + section/clause id) per answer.\<br>i|

|AC-03|Retrieval combines lexical (BM25) and semantic search and fuses the two result sets (e.g., Reciprocal\<br>Rank Fusion)beforegeneration.|

|AC-04|Retrieved candidates are reranked (cross-encoder or hosted reranker) before the top-K is passed to\<br>thegenerator.\<br>l  i|

|AC-05|When the corpus does not support an answer, the system abstains or flags low confidence rather\<br>than fabricating.\<br>i|

|AC-06|Answers are returned as a validated structured object (Pydantic / JSON schema) containing answer\<br>text, citations, applicable rule / policy clause, any threshold or timeline, and the required action, and\<br>agrounding /confidence indicator.\<br>i          i|

|AC-07|Multi-part or ambiguous queries are transformed (rewrite / expansion / decomposition) before\<br>retrieval.\<br>i     i        t|

|AC-08|A golden evaluation set of ≥ 20 questions with reference answers / expected contexts is committed\<br>with a re-runnable scoringscript.|

|AC-09|RAGAS metrics (context precision, context recall, faithfulness, answer relevancy) are computed and\<br>the numeric results committed as a report artifact.|

|AC-10|At least two candidate LLMs are evaluated on the custom eval set and a comparison (metrics +\<br>selection rationale)is committed.|





**### \*\*5.2 Non-Functional Requirements\*\*** 

|**\*\*ID\*\***|**\*\*Requirement\*\***\<br>t ii|

\|---|---|

|NFR-01|No secrets or API keys committed; configuration via environment variables with a\<br>committed .env.example.\<br>i         t|

|NFR-02|Pipeline and evaluation run end-to-end from a single documented command with committed sample\<br>data and a READMEquick-start.\<br>i    i      t|

|NFR-03|All data is synthetic; any PII is synthetic, masked where shown, and never written to logs in plaintext,\<br>per a documentedpolicy.\<br>ti|

|NFR-04|Retrieval parameters (chunk size / overlap, top-K, thresholds, reranker settings) are externalized in\<br>config,not hard-coded inline.|

|NFR-05|Provider / model calls implement basic retries and fail gracefully — a clear error or safe fallback\<br>response on repeated failure — rather than crashing.\<br>t|

|NFR-06|Every quality claim is reproducible — the eval script and dataset are committed so metrics can be re-\<br>derived.\<br>l|

|NFR-07|Cost and latency are addressed at concept level only: briefly note the approximate cost / latency of a\<br>representativequery (no benchmarkingorperformancegovernance required).|

|NFR-08|Basic logging of queries and answers is in place; no answer is returned without provenance above the\<br>abstention threshold.|





**## \*\*6. Functional Scope\*\*** 

**### \*\*6.1 In Scope\*\*** 

\- Corpus ingestion, cleaning, and chunking (fixed / recursive / semantic / sentence-window — your choice, justified). 

\- Embedding, indexing, and metadata design (document type, section / clause id) in a persisted vector store. 

\- Hybrid retrieval (BM25 + semantic) with fusion and a reranking stage. 

\- Query transformation and grounded generation with clause-level citations, applicable rule / policy clause, any threshold or timeline, and the required action extraction, and abstention. 

\- Structured output, an evaluation harness (RAGAS + golden set), and a two-model comparison. 

\- A minimal interface (CLI / Streamlit / Gradio / FastAPI) to exercise the pipeline. 

**### \*\*6.2 Out of Scope\*\*** 

\- Real exchange connectivity, order routing, or trade execution. 

\- Real customer or confidential data of any kind. 

\- Fine-tuning or training a model; building your own vector database engine. 

\- Production deployment, multi-region, authentication, and user management; front-end visual polish. 

**## \*\*7. Implementation Expectations\*\*** 

The rubric scores committed evidence. Each category below states what must exist in the repository. Marks per category are shown in Section 9. 

**### \*\*7.1 Business & Requirements (15 marks)\*\*** 

\- docs/business-case.md covering problem, target users, corpus description, domain guardrails, and success metrics. 

\- Acceptance criteria in testable form (AC-NN) under specs/ for at least the core retrieval and grounding behaviors. 

\- A documented guardrail policy: citation-required, abstention behavior, information-only (no trading recommendation or investment advice), and out-of-scope handling. 

**### \*\*7.2 Ingestion & Indexing (15 marks)\*\*** 

\- An ingestion module with an explicit, justified chunking strategy (not naive fixed-size only). 

\- An embedding-model selection note (MTEB-informed rationale, dimensionality / cost trade-offs). 

\- A populated, persisted vector store with a sensible metadata schema (source doc, section / clause id, doc type). 

**### \*\*7.3 Retrieval Quality (25 marks)\*\*** 

\- Hybrid retrieval combining BM25 and semantic search, with fusion (RRF or equivalent). 

\- A reranking stage (cross-encoder or hosted reranker) selecting the final top-K. 

\- Query transformation (rewrite / expansion / decomposition) for complex, multi-part questions. 

\- Retrieval configuration surfaced (top-K, thresholds, chunk params) in config files. 

**### \*\*7.4 Generation & Grounding (15 marks)\*\*** 

\- Clause-level citations on every answer; committed sample outputs demonstrating provenance. 

\- Faithfulness / anti-hallucination controls: grounding prompt and abstention path, with a committed refusal example. 

\- Validated structured output (Pydantic / JSON schema) for answer + citations + applicable rule / policy clause, any threshold or timeline, and the required action + confidence. 

**### \*\*7.5 RAG Evaluation (20 marks)\*\*** 

\- A committed golden eval set (≥ 20 Q with reference answers / expected contexts). 

\- Committed RAGAS metric results (context precision, context recall, faithfulness, answer relevancy). 

\- A failure-taxonomy analysis distinguishing retrieval vs grounding vs synthesis failures. 

\- A re-runnable, deterministic eval harness so metrics can be regenerated. 

**### \*\*7.6 Model Selection & Engineering (10 marks)\*\*** 

\- A committed comparison of ≥ 2 candidate LLMs on the custom eval set with a selection rationale. 

\- Engineering hygiene: README quick-start, single-command run, config externalized, no secrets, basic retries. 

\- A brief cost / latency note for a representative query (concept level). 

**## \*\*8. Expected Outcomes & Deliverables\*\*** 

By the end of 15 hours, the submitted Git repository must contain the Mandatory items below. Good-to-Have items differentiate Merit and Distinction submissions. 

**### \*\*8.1 Mandatory\*\*** 

\- Working RAG pipeline runnable locally via a single command, with committed synthetic corpus and README quick-start. 

\- docs/business-case.md — problem, users, corpus, guardrails, success metrics. 

\- Ingestion + indexing module; hybrid retrieval + fusion + reranking; grounded generation with clause-level citations, applicable rule / policy clause, any threshold or timeline, and the required action extraction, and abstention; structured output schema. 

\- Golden eval set + re-runnable eval script; committed RAGAS metric report; two-model comparison report. 

\- Specs with AC-NN acceptance criteria and AC-referenced tests/eval entries; .env.example; no committed secrets. 

\- PR-driven Git history: at least 3 PR-driven merges (git merge --no-ff); no direct pushes to main. 

**### \*\*8.2 Good-to-Have\*\*** 

\- Agentic / corrective retrieval (self-RAG or corrective-RAG) with adaptive fallback. 

\- Product-aware routing (cash vs F&O vs currency) before retrieval. 

\- A retrieval-regression gate that fails when RAGAS scores drop below a threshold. 

\- A lightweight UI (Streamlit / Gradio) with citation highlighting. 

**## \*\*9. Evaluation Rubric\*\*** 

21 parameters across 6 categories, 100 marks. Scoring is model-portable (Google Gemini — not Claude) and reads only committed repository evidence. Numeric-threshold parameters (marked Deterministic) are scored in Python with no model judgment. 

|**\*\*Category\*\***|**\*\*Parameters(max)\*\***\<br>ii|**\*\*Marks\*\***|

\|---|---|---|

|**\*\*Business & Requirements\*\***\<br>**\*\*i\*\***|Business-case clarity (5) · AC definition (5) · Domain rules & guardrails\<br>(5)\<br>i    i|15|

|**\*\*Ingestion & Indexing\*\***|Chunking strategy + rationale (6) · Embedding selection (4) · Vector\<br>store & indexing (5)|15|

|**\*\*Retrieval Quality\*\***\<br>**\*\*i\*\***|Hybrid search (7) · RRF fusion (4) · Reranking (7) · Query\<br>transformation(4)· Retrieval config (3)\<br>i     i|25|

|**\*\*Generation & Grounding\*\***\<br>**\*\*i\*\***|Clause-level citations (6) · Faithfulness / abstention (5) · Structured\<br>output(4)\<br>i|15|

|**\*\*RAG Evaluation\*\***\<br>**\*\*i\*\***|Golden set (5) · RAGAS metrics — Deterministic (8) · Failure\<br>taxonomy (4)· Reproducible harness(3)\<br>i|20|

|**\*\*Model Selection &\*\***\<br>**\*\*Engineering\*\***|≥ 2 LLMs compared — Deterministic (4) · Reproducibility / secrets (4)\<br>· Cost & latency (2)|10|

|**\*\*TOTAL\*\***|21parameters|100|





|**\*\*Grade Band\*\***|**\*\*Range\*\***|

\|---|---|

|Distinction|≥ 90|

|ii\<br>Merit|75 – 89|

|Pass|60 – 74|

|Not Yet Passed|< 60|





*\_On completion you will have demonstrated the skill industry is seeking: designing a production-grade, measurable RAG pipeline that powers a brokerage-rules and trading-policy assistant — with hybrid retrieval, reranking, grounded citations, and evidence-backed quality — not a prompt-and-hope prototype.\_* 