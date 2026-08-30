from sentence_transformers import CrossEncoder
from config.config_manager import ConfigManager

class CrossEncoderReranker:
    """
    Reranks retrieved candidate chunks relative to a query using a local Cross-Encoder.
    This provides deeper contextual scoring compared to single-vector search.
    """
    def __init__(self, config_manager: ConfigManager = None, local_overrides: dict = None):
        self.config_manager = config_manager or ConfigManager()
        
        # Load configuration sections from global config
        rerank_cfg = self.config_manager.get_section("reranker")
        retrieval_cfg = self.config_manager.get_section("retrieval")
        
        # Extract configurations directly into distinct properties (no self.config dict lookup)
        self.model_name = local_overrides.get("model_name") if local_overrides and "model_name" in local_overrides else rerank_cfg.get("model_name")
        self.rerank_top_k = local_overrides.get("rerank_top_k") if local_overrides and "rerank_top_k" in local_overrides else retrieval_cfg.get("rerank_top_k")
            
        # Initialize CrossEncoder model lazily
        self.model = None

    def _lazy_init(self):
        if self.model is None:
            print(f"Loading CrossEncoder reranker model: {self.model_name}...")
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        """
        Takes a query and a list of retrieved candidate chunks,
        computes cross-encoder scores, sorts them, and returns the top-K.
        """
        if not candidates:
            return []
            
        self._lazy_init()
        
        if top_k is None:
            top_k = int(self.rerank_top_k)
            
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
