import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.generation.schemas import GroundedAnswer

# Initialize FastAPI application
app = FastAPI(
    title="Brokerage Rules & Trading Policy Assistant API",
    description="API backend for querying synthetic brokerage policies and exchange rules.",
    version="1.0.0"
)

# Global Configuration and Class Instances
config_manager = ConfigManager()
pipeline = IngestionPipeline(config_manager)
rag_pipeline = RAGPipeline(config_manager)

class QueryRequest(BaseModel):
    query: str

class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int

class HealthResponse(BaseModel):
    status: str

@app.post("/query", response_model=GroundedAnswer)
def run_query(request: QueryRequest):
    """
    Executes the hybrid RAG pipeline using the unified orchestrator:
    1. Transforms/expands query using Gemini.
    2. Fetches candidates using RRF lexical + vector search.
    3. Reranks candidates using local Cross-Encoder.
    4. Generates grounded answer using Gemini structured output.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        grounded_answer, _ = rag_pipeline.run_query(query)
        return grounded_answer
    except Exception as e:
        print(f"Error processing query API call: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
def run_ingest():
    """
    Executes document ingestion. Re-creates persistent indexes.
    Idempotent and re-runnable.
    """
    try:
        num_chunks = pipeline.run()
        
        # Reload the new FAISS index from disk in RAGPipeline retriever
        rag_pipeline.retriever.reload_db()
        
        return IngestResponse(status="success", chunks_ingested=num_chunks)
    except Exception as e:
        print(f"Error running ingestion API call: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def check_health():
    """
    Standard API health check.
    """
    return HealthResponse(status="healthy")

import gradio as gr
from src.interface.app import demo

# Mount Gradio frontend UI directly onto the FastAPI application at root path
app = gr.mount_gradio_app(app, demo, path="/")

def start_server():
    """Starts the unified FastAPI + Gradio server."""
    # Ensure retriever loads the index that was built by pipeline.run() in main.py
    rag_pipeline.retriever.reload_db()
    
    api_cfg = config_manager.get_section("api")
    host = api_cfg.get("host", "127.0.0.1")
    port = int(api_cfg.get("port", 8000))
    print(f"Launching Unified FastAPI + Gradio Server on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
