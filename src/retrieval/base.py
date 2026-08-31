from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """
    Abstract base class for all RAG retrieval strategies.
    Defines a unified retrieval interface.
    """
    @abstractmethod
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieves candidate documents matching the query.
        Returns a list of dicts with keys: id, document, metadata, score.
        """
        pass
