import chromadb
from config.config_manager import ConfigManager

class SemanticRetriever:
    """
    Encapsulates dense vector semantic search against the Chroma collection.
    """
    def __init__(self, config_manager: ConfigManager, client: chromadb.PersistentClient, config: dict = None):
        self.config_manager = config_manager
        self.client = client
        
        if config is None:
            # Load configuration sections from global config
            retrieval_cfg = self.config_manager.get_section("retrieval")
            vstore_cfg = self.config_manager.get_section("vector_store")
            self.config = {
                "top_k_semantic": retrieval_cfg.get("top_k_semantic"),
                "collection_name": vstore_cfg.get("collection_name")
            }
        else:
            self.config = config
            
        self.collection_name = self.config["collection_name"]

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
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
                dist = results["distances"][0][idx]
                retrieved.append({
                    "id": results["ids"][0][idx],
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "score": float(1.0 / (1.0 + dist))
                })
        return retrieved
