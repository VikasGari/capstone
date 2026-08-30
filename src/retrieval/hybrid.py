import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from config.config_manager import ConfigManager

class HybridRetriever:
    """
    Combines Vector semantic search and BM25 lexical search.
    Fuses rankings using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Local defaults
        local_defaults = {
            "persist_directory": "data/chroma_db",
            "collection_name": "trading_policy_collection",
            "top_k_semantic": 10,
            "top_k_bm25": 10,
            "rrf_k": 60
        }
        
        # Merge with global overrides (Global config has higher precedence)
        vstore_cfg = self.config_manager.get_section("vector_store")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        self.config = local_defaults.copy()
        
        if "persist_directory" in vstore_cfg:
            self.config["persist_directory"] = vstore_cfg["persist_directory"]
        if "collection_name" in vstore_cfg:
            self.config["collection_name"] = vstore_cfg["collection_name"]
        if "top_k_semantic" in retrieval_cfg:
            self.config["top_k_semantic"] = retrieval_cfg["top_k_semantic"]
        if "top_k_bm25" in retrieval_cfg:
            self.config["top_k_bm25"] = retrieval_cfg["top_k_bm25"]
        if "rrf_k" in retrieval_cfg:
            self.config["rrf_k"] = retrieval_cfg["rrf_k"]
            
        if local_overrides:
            self.config.update(local_overrides)
            
        # Initialize chroma client and get collection details
        self.client = chromadb.PersistentClient(path=self.config["persist_directory"])
        self.collection_name = self.config["collection_name"]
        
        # Setup lazy initialization of BM25
        self.bm25 = None
        self.corpus_chunks = []
        
    def _initialize_bm25(self):
        """Loads all documents from Chroma collection and fits BM25."""
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            print(f"Error fetching collection '{self.collection_name}'. Has ingestion run? {e}")
            return
            
        # Get all entries (documents, metadatas, ids)
        results = collection.get(include=["documents", "metadatas"])
        if not results or not results["documents"]:
            print(f"Warning: Chroma collection '{self.collection_name}' is empty. Cannot initialize BM25.")
            return
            
        self.corpus_chunks = []
        tokenized_corpus = []
        
        for idx, doc in enumerate(results["documents"]):
            chunk_data = {
                "id": results["ids"][idx],
                "document": doc,
                "metadata": results["metadatas"][idx]
            }
            self.corpus_chunks.append(chunk_data)
            
            # Simple tokenization for BM25 (lowercased alphanumeric tokens)
            tokens = self._tokenize(doc)
            tokenized_corpus.append(tokens)
            
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 retriever successfully initialized on {len(self.corpus_chunks)} chunks.")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer that converts text to alphanumeric lowercase tokens."""
        return re.findall(r"\w+", text.lower())

    def retrieve_semantic(self, query: str, top_k: int = None) -> list[dict]:
        """Queries the Chroma collection using semantic search."""
        if top_k is None:
            top_k = int(self.config["top_k_semantic"])
            
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            print(f"Error getting collection: {e}")
            return []
            
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            for idx in range(len(results["documents"][0])):
                # Chroma returns 'distances' (usually L2 distance for cosine/L2 metric).
                # Convert L2 distance or cosine distance to score if needed.
                dist = results["distances"][0][idx]
                retrieved.append({
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "score": float(1.0 / (1.0 + dist)) # Map distance to a pseudo-similarity score
                })
        return retrieved

    def retrieve_bm25(self, query: str, top_k: int = None) -> list[dict]:
        """Queries the corpus using BM25 lexical search."""
        if self.bm25 is None:
            self._initialize_bm25()
            
        if self.bm25 is None:
            return []
            
        if top_k is None:
            top_k = int(self.config["top_k_bm25"])
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and select top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0: # Avoid returning entirely irrelevant matches
                continue
            chunk = self.corpus_chunks[idx]
            retrieved.append({
                "id": chunk["id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"],
                "score": float(score)
            })
        return retrieved

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
