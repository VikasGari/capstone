from config.config_manager import ConfigManager
from src.retrieval.transformer import QueryTransformer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import GroundedGenerator
from src.generation.schemas import GroundedAnswer

class RAGPipeline:
    """
    Unified RAG execution pipeline orchestrator.
    Encapsulates query transformation, hybrid retrieval, cross-encoder reranking,
    and grounded generation with structured Pydantic schemas.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.transformer = QueryTransformer(self.config_manager)
        self.retriever = HybridRetriever(self.config_manager)
        self.reranker = CrossEncoderReranker(self.config_manager)
        self.generator = GroundedGenerator(self.config_manager)

    def run_query(self, query: str, model_name: str = None) -> tuple[GroundedAnswer, list[dict]]:
        """
        Executes the full RAG pipeline over a query.
        Returns:
            tuple: (GroundedAnswer pydantic response, list of reranked reference context chunks)
        """
        # Step 1: Query Transformation (Expansion)
        sub_queries = [query]
        if self.transformer.chain:
            try:
                sub_queries = self.transformer.transform(query)
            except Exception as e:
                print(f"Warning: Query transformer failed: {e}. Falling back to raw query.")
                
        # Step 2: Retrieve candidates for all expanded queries using Hybrid RRF
        all_candidates = []
        seen_ids = set()
        for sq in sub_queries:
            candidates = self.retriever.retrieve(sq)
            for cand in candidates:
                if cand["id"] not in seen_ids:
                    seen_ids.add(cand["id"])
                    all_candidates.append(cand)
                    
        # Step 3: Local Cross-Encoder Reranking
        reranked = self.reranker.rerank(query, all_candidates)
        
        # Step 4: Grounded Answer Generation
        ans_obj = self.generator.generate(query, reranked, model_name=model_name)
        
        return ans_obj, reranked
