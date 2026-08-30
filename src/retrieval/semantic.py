import chromadb

class SemanticRetriever:
    """
    Encapsulates dense vector semantic search against the Chroma collection.
    Decoupled from ConfigManager: accepts parameters directly in constructor.
    """
    def __init__(self, client: chromadb.PersistentClient, collection_name: str, top_k_semantic: int):
        self.client = client
        self.collection_name = collection_name
        self.top_k_semantic = int(top_k_semantic)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Queries the Chroma collection using semantic search."""
        if top_k is None:
            top_k = self.top_k_semantic
            
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
                dist = results["distances"][0][idx]
                retrieved.append({
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "score": float(1.0 / (1.0 + dist))
                })
        return retrieved
