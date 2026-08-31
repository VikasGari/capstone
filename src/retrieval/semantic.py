from langchain_community.vectorstores import FAISS
from src.retrieval.base import BaseRetriever
from src.helpers.math_utils import l2_to_similarity

class SemanticRetriever(BaseRetriever):
    """
    Encapsulates dense vector semantic search against the FAISS index.
    Decoupled from ConfigManager: accepts parameters directly in constructor.
    """
    def __init__(self, db: FAISS, top_k_semantic: int):
        self.db = db
        self.top_k_semantic = int(top_k_semantic)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Queries the FAISS index using semantic similarity search."""
        if self.db is None:
            print("Warning: FAISS database index is not initialized.")
            return []
            
        if top_k is None:
            top_k = self.top_k_semantic
            
        try:
            results = self.db.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            print(f"Error querying FAISS: {e}")
            return []
            
        retrieved = []
        for doc, dist in results:
            retrieved.append({
                "id": doc.metadata.get("id", ""),
                "document": doc.page_content,
                "metadata": doc.metadata,
                "score": l2_to_similarity(dist)
            })
        return retrieved
