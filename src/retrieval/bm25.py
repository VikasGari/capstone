import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from src.retrieval.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    """
    Encapsulates lexical search utilizing the BM25Okapi ranking model.
    Decoupled from ConfigManager: accepts parameters directly in constructor.
    """
    def __init__(self, db: FAISS, top_k_bm25: int):
        self.db = db
        self.top_k_bm25 = int(top_k_bm25)
        self.bm25 = None
        self.corpus_chunks = []

    def _initialize_bm25(self):
        """Loads all documents from FAISS index and fits BM25."""
        if self.db is None:
            print("Warning: FAISS database index is not initialized. Cannot initialize BM25.")
            return
            
        try:
            docstore = self.db.docstore._dict
        except Exception as e:
            print(f"Error accessing FAISS docstore: {e}")
            return
            
        if not docstore:
            print("Warning: FAISS docstore is empty. Cannot initialize BM25.")
            return
            
        self.corpus_chunks = []
        tokenized_corpus = []
        
        for doc_id, doc in docstore.items():
            chunk_data = {
                "id": doc.metadata.get("id", doc_id),
                "document": doc.page_content,
                "metadata": doc.metadata
            }
            self.corpus_chunks.append(chunk_data)
            
            tokens = self._tokenize(doc.page_content)
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
