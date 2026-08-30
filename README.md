# 📈 Brokerage Rules & Trading Policy Assistant

A grounded Retrieval-Augmented Generation (RAG) assistant for Support and Operations staff, providing accurate answers about trading rules, F&O margins, fees, and settlement timelines. It extracts compliance-grounded answers with clause-level citations from synthetic exchange policy rulebooks.

---

## 🚀 Quick Start (Single-Command Launch)

The application starts both the ingestion pipeline and the unified web frontend + backend API server from a single command.

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

### 3. Launch Unified Server
Run the main script to trigger document ingestion and launch the server:
```powershell
python main.py
```
* **Ingestion:** Automatically indexes exchange rulebooks and brokerage margin policies into a persisted Chroma vector index (idempotent).
* **Unified Service:** Launches a Uvicorn process hosting the FastAPI endpoints (e.g. `/query`, `/health`, `/ingest`) and mounts the interactive Gradio UI directly at the root path.
* **Access URL:** Open your browser and navigate to **`http://localhost:8000/`**.

---

## 📊 Evaluation & Model Comparisons

To run RAGAS metric scoring (Faithfulness, Answer Relevancy, Recall, and Precision) on the golden dataset and compile model performance comparisons:

```powershell
python run_eval.py
```
This writes simple, structured reports to the following paths:
* `docs/eval_report.md`: Grounding metrics and failure taxonomy.
* `docs/model_comparison.md`: LLM selection justification.

---

## 🧪 Running Unit Tests

Run the full pytest suite to verify RAG configuration, ingestion, retrievers, and structured generators:
```powershell
pytest
```

---

## 📂 Project Directory Layout

* `config/`: Contains `config.yaml` and the `ConfigManager` configuration loader.
* `data/`: Stores synthetic policy text source documents, `golden_set.json` evaluation queries, and persistent Chroma database files.
* `docs/`: Holds technical Rationales, Business Cases, and Evaluation Reports.
* `src/`: Package code modules:
  * `ingestion/`: Idempotent document text ingestion pipelines.
  * `retrieval/`: Decoupled BM25, Semantic search, and Hybrid RRF fusion.
  * `generation/`: Modular Pydantic schemas, prompts, and generator clients.
  * `interface/`: Unified FastAPI and Gradio application files.
* `tests/`: Automated pytest unit test suite.
