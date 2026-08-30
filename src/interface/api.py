import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.transformer import QueryTransformer
from src.generation.generator import GroundedGenerator, GroundedAnswer

# Initialize FastAPI application
app = FastAPI(
    title="Brokerage Rules & Trading Policy Assistant API",
    description="API backend for querying synethic brokerage policies and exchange rules.",
    version="1.0.0"
)

# Global Configuration and Class Instances
config_manager = ConfigManager()
pipeline = IngestionPipeline(config_manager)
retriever = HybridRetriever(config_manager)
reranker = CrossEncoderReranker(config_manager)
transformer = QueryTransformer(config_manager)
generator = GroundedGenerator(config_manager)

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
    Executes the hybrid RAG pipeline:
    1. Transforms/expands query using Gemini.
    2. Fetches candidates using RRF lexical + vector search.
    3. Reranks candidates using local Cross-Encoder.
    4. Generates grounded answer using Gemini structured output.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Step 1: Query Transformation
        sub_queries = transformer.transform(query)
        
        # Step 2: Retrieve candidates for all expanded queries
        all_candidates = []
        seen_ids = set()
        
        for sq in sub_queries:
            # retrieve combines BM25 & Semantic vector search using RRF
            candidates = retriever.retrieve(sq)
            for cand in candidates:
                if cand["id"] not in seen_ids:
                    seen_ids.add(cand["id"])
                    all_candidates.append(cand)
                    
        # Step 3: Reranking
        reranked = reranker.rerank(query, all_candidates)
        
        # Step 4: Grounded Generation
        grounded_answer = generator.generate(query, reranked)
        return grounded_answer
        
    except Exception as e:
        print(f"Error processing query API call: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
def run_ingest():
    """
    Executes document ingestion. Re-creates Chroma persistent indexes.
    Idempotent and re-runnable.
    """
    try:
        num_chunks = pipeline.run()
        
        # Force BM25 retriever to reload the index next time it is queried
        retriever.bm25 = None
        retriever.corpus_chunks = []
        
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

def start_server():
    """Starts the FastAPI backend server."""
    api_cfg = config_manager.get_section("api")
    host = api_cfg.get("host", "127.0.0.1")
    port = int(api_cfg.get("port", 8000))
    print(f"Launching FastAPI Server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
