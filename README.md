# 📈 Brokerage Rules & Trading Policy Assistant

A grounded Retrieval-Augmented Generation (RAG) assistant for Support and Operations staff, providing accurate answers about trading rules, F&O margins, fees, and settlement timelines. It extracts compliance-grounded answers with clause-level citations from synthetic exchange policy rulebooks.

---

## 🚀 Quick Start (Single-Command Entrypoint)

The application provides a unified single CLI entrypoint (`main.py`) for running the server, executing evaluation comparisons, triggering ingestion, and running CLI queries.

### 1. Installation & Environment Setup

Clone this repository to your workspace, create a virtual environment, and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials

Create a `.env` file in the root directory (based on `.env.example`) and fill in your Gemini API key:
```ini
GEMINI_API_KEY="your-google-gemini-api-key-here"
```

### 3. Launch Modes (`main.py`)

#### A. Launch Server & Frontend (Default Mode)
Runs automated ingestion checks and starts both the FastAPI REST endpoints and interactive Gradio UI:
```powershell
python main.py
```
* **Ingestion:** Automatically indexes exchange rulebooks and brokerage margin policies into a persisted FAISS vector index (idempotent).
* **Unified Service:** Launches Uvicorn hosting FastAPI endpoints (`/query`, `/health`, `/ingest`) and mounts the interactive Gradio UI.
* **Access URL:** Open your browser and navigate to **`http://localhost:8000/`**.

#### B. Run RAGAS Evaluation Comparison
Executes comparative scoring (Faithfulness, Answer Relevancy, Recall, Precision) over the golden set using rate-limiting guardrails:
```powershell
python main.py --eval
```
* Generates the consolidated report at [`docs/eval_report.md`](file:///d:/VS%20Code/Capstone/docs/eval_report.md).

#### C. Ingest Documents Only
Re-indexes policy documents in `data/corpus/` into FAISS:
```powershell
python main.py --ingest
```

#### D. Run Direct CLI Query
Queries the RAG pipeline directly from the command line:
```powershell
python main.py --query "What is the pre-open session order entry window?"
```

---

## 📊 Configuration Centralization (`config/config.yaml`)

All system parameters are controlled strictly via [`config/config.yaml`](file:///d:/VS%20Code/Capstone/config/config.yaml):
* **Embedding & Reranker Models:** Open-source local models (`sentence-transformers/all-MiniLM-L6-v2` & `cross-encoder/ms-marco-MiniLM-L-6-v2`).
* **LLM Selection & Fallbacks:** Primary model (`gemini-3.5-flash-lite`) with LCEL structured fallback (`gemini-3.1-flash-lite`).
* **Rate Limiting Guardrails:** `rate_limit_delay_seconds` (e.g. `4.0` for 15 RPM limit, or `null` to disable).
* **Refusal & Disclaimer Text:** Externalized compliance domain refusal and legal disclaimer prompts.

---

## 🧪 Running Unit Tests

Run the full pytest suite to verify RAG configuration, ingestion, retrievers, and structured generators:
```powershell
pytest
```

---

## 📂 Project Directory Layout

* `config/`: Contains `config.yaml` and the `ConfigManager` configuration loader.
* `data/`: Stores synthetic policy text source documents, `golden_set.json` evaluation queries, and persistent FAISS index files.
* `docs/`: Holds technical specifications ([`acceptance-criteria.md`](file:///d:/VS%20Code/Capstone/docs/acceptance-criteria.md)), evaluation reports ([`eval_report.md`](file:///d:/VS%20Code/Capstone/docs/eval_report.md)), and walkthroughs.
* `src/`: Package code modules (all files $\le 150$ lines):
  * `ingestion/`: Idempotent document text ingestion & clause parser.
  * `retrieval/`: Decoupled BM25, Semantic search, Hybrid RRF fusion, and Cross-Encoder reranker.
  * `generation/`: Modular Pydantic schemas, LangChain LCEL prompts, and grounded generator clients.
  * `evaluation/`: RAGAS evaluation harness and LLM judge.
  * `interface/`: Unified FastAPI and Gradio web interface.
* `tests/`: Automated pytest unit test suite.
