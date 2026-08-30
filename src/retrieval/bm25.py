import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    Encapsulates lexical search utilizing the BM25Okapi ranking model.
    Decoupled from ConfigManager: accepts parameters directly in constructor.
    """
    def __init__(self, client: chromadb.PersistentClient, collection_name: str, top_k_bm25: int):
        self.client = client
        self.collection_name = collection_name
        self.top_k_bm25 = int(top_k_bm25)
        self.bm25 = None
        self.corpus_chunks = []

    def _initialize_bm25(self):
        """Loads all documents from Chroma collection and fits BM25."""
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            print(f"Error fetching collection '{self.collection_name}'. Has ingestion run? {e}")
            return
            
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
            
            tokens = self._tokenize(doc)
            tokenized_corpus.append(tokens)
            
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 retriever successfully initialized on {len(self.corpus_chunks)} chunks.")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer that converts text to alphanumeric lowercase tokens."""
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Queries the corpus using BM25 lexical search."""
        if self.bm25 is None:
            self._initialize_bm25()
            
        if self.bm25 is None:
            return []
            
        if top_k is None:
            top_k = self.top_k_bm25
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            chunk = self.corpus_chunks[idx]
            retrieved.append({
                "id": chunk["id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"],
                "score": float(score)
            })
        return retrieved
