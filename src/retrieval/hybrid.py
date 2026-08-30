import chromadb
from config.config_manager import ConfigManager
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.semantic import SemanticRetriever

class HybridRetriever:
    """
    Orchestrates Vector semantic search and BM25 lexical search.
    Fuses candidate rankings using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration sections directly from global config
        vstore_cfg = self.config_manager.get_section("vector_store")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        self.config = {
            "persist_directory": vstore_cfg.get("persist_directory"),
            "collection_name": vstore_cfg.get("collection_name"),
            "top_k_semantic": retrieval_cfg.get("top_k_semantic"),
            "top_k_bm25": retrieval_cfg.get("top_k_bm25"),
            "rrf_k": retrieval_cfg.get("rrf_k")
        }
        
        if local_overrides:
            self.config.update(local_overrides)
            
        # Initialize the persistent client
        self.client = chromadb.PersistentClient(path=self.config["persist_directory"])
        
        # Instantiate subcomponents, passing the resolved config containing any overrides
        self.bm25_retriever = BM25Retriever(self.config_manager, self.client, self.config)
        self.semantic_retriever = SemanticRetriever(self.config_manager, self.client, self.config)

    # Expose helper to fit index (used in API and evaluation harness)
    def _initialize_bm25(self):
        self.bm25_retriever._initialize_bm25()

    @property
    def bm25(self):
        return self.bm25_retriever.bm25
        
    @bm25.setter
    def bm25(self, value):
        self.bm25_retriever.bm25 = value
        
    @property
    def corpus_chunks(self):
        return self.bm25_retriever.corpus_chunks
        
    @corpus_chunks.setter
    def corpus_chunks(self, value):
        self.bm25_retriever.corpus_chunks = value

    def retrieve_semantic(self, query: str, top_k: int = None) -> list[dict]:
        """Delegates semantic retrieval to the SemanticRetriever subcomponent."""
        return self.semantic_retriever.retrieve(query, top_k)

    def retrieve_bm25(self, query: str, top_k: int = None) -> list[dict]:
        """Delegates lexical retrieval to the BM25Retriever subcomponent."""
        return self.bm25_retriever.retrieve(query, top_k)

    def retrieve(self, query: str, top_k_semantic: int = None, top_k_bm25: int = None, rrf_k: int = None) -> list[dict]:
        """
        Performs hybrid search by combining semantic and lexical BM25 results,
        fusing the ranks using Reciprocal Rank Fusion (RRF).
        """
        semantic_results = self.retrieve_semantic(query, top_k_semantic)
        bm25_results = self.retrieve_bm25(query, top_k_bm25)
        
        if rrf_k is None:
            rrf_k = int(self.config["rrf_k"])
            
        # RRF formula: Score(d) = sum(1 / (rrf_k + rank_r(d)))
        rrf_scores = {}
        doc_details = {} # Store metadata and document text map
        
        # Helper to process results and assign rank-based RRF scores
        def add_rrf_scores(results):
            for rank, item in enumerate(results):
                doc_id = item["id"]
                if doc_id not in doc_details:
                    doc_details[doc_id] = {
                        "id": doc_id,
                        "document": item["document"],
                        "metadata": item["metadata"]
                    }
                # RRF score addition
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
                
        add_rrf_scores(semantic_results)
        add_rrf_scores(bm25_results)
        
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_results = []
        for doc_id in sorted_ids:
            doc = doc_details[doc_id]
            fused_results.append({
                "id": doc["id"],
                "document": doc["document"],
                "metadata": doc["metadata"],
                "score": rrf_scores[doc_id]
            })
            
        return fused_results
