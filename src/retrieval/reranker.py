from sentence_transformers import CrossEncoder
from config.config_manager import ConfigManager

class CrossEncoderReranker:
    """
    Reranks retrieved candidate chunks relative to a query using a local Cross-Encoder.
    This provides deeper contextual scoring compared to single-vector search.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Fetch configuration directly from global config
        rerank_cfg = self.config_manager.get_section("reranker")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        self.config = {
            "model_name": rerank_cfg.get("model_name"),
            "rerank_top_k": retrieval_cfg.get("rerank_top_k")
        }
            
        if local_overrides:
            self.config.update(local_overrides)
            
        # Initialize CrossEncoder model lazily
        self.model = None

    def _lazy_init(self):
        if self.model is None:
            model_name = self.config["model_name"]
            print(f"Loading CrossEncoder reranker model: {model_name}...")
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        """
        Takes a query and a list of retrieved candidate chunks,
        computes cross-encoder scores, sorts them, and returns the top-K.
        """
        if not candidates:
            return []
            
        self._lazy_init()
        
        if top_k is None:
            top_k = int(self.config["rerank_top_k"])
            
        # Format inputs for Cross-Encoder: list of (query, document) pairs
        pairs = [(query, c["document"]) for c in candidates]
        
        # Predict scores (higher is more relevant)
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        reranked = []
        for idx, score in enumerate(scores):
            cand = candidates[idx].copy()
            cand["rerank_score"] = float(score)
            reranked.append(cand)
            
        # Sort by rerank score descending
        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
