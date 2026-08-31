import tempfile
from pathlib import Path
from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

def test_hybrid_retrieval_and_reranking():
    # Setup temporary directories for testing
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as corpus_dir, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as persist_dir:
        corpus_path = Path(corpus_dir)
        
        # Create two sample files
        doc1 = corpus_path / "doc1.txt"
        with open(doc1, "w", encoding="utf-8") as f:
            f.write("DOCUMENT ID: DOC_1\n")
            f.write("DOCUMENT TITLE: Trading Limit Policy\n")
            f.write("CATEGORY: Margin Policy\n")
            f.write("="*40 + "\n")
            f.write("Clause 1.1: Cash Limit\n")
            f.write("Clients must maintain cash balance above 1000 currency units at all times.\n")
            
        doc2 = corpus_path / "doc2.txt"
        with open(doc2, "w", encoding="utf-8") as f:
            f.write("DOCUMENT ID: DOC_2\n")
            f.write("DOCUMENT TITLE: Settlement Timelines\n")
            f.write("CATEGORY: Settlement\n")
            f.write("="*40 + "\n")
            f.write("Clause 2.1: Pay Out\n")
            f.write("Payout of funds will complete on T+1 day by 14:00 hours.\n")
            
        # Add 8 dummy documents to ensure BM25 IDF works correctly (requires N > 2*n(q))
        for i in range(3, 11):
            dummy_doc = corpus_path / f"doc{i}.txt"
            with open(dummy_doc, "w", encoding="utf-8") as f:
                f.write(f"DOCUMENT ID: DOC_{i}\n")
                f.write(f"DOCUMENT TITLE: General Policy {i}\n")
                f.write("CATEGORY: General\n")
                f.write("="*40 + "\n")
                f.write(f"Clause {i}.1: Section {i}\n")
                f.write(f"This is a dummy document number {i} to increase corpus size for lexical BM25 retrieval.\n")
                
        # Ingest documents
        config_manager = ConfigManager()
        config_manager.config["vector_store"]["persist_directory"] = persist_dir
        config_manager.config["paths"]["corpus_directory"] = corpus_dir
        
        pipeline = IngestionPipeline(config_manager=config_manager)
        pipeline.run()
        
        # Initialize HybridRetriever
        retriever = HybridRetriever(config_manager=config_manager)
        
        # 1. Test Semantic Search
        semantic_results = retriever.retrieve_semantic("When does payout of funds complete?", top_k=2)
        assert len(semantic_results) > 0
        assert any("T+1" in r["document"] for r in semantic_results)
        
        # 2. Test BM25 Search
        bm25_results = retriever.retrieve_bm25("cash balance 1000", top_k=2)
        assert len(bm25_results) > 0
        assert any("DOC_1" in r["metadata"]["doc_id"] for r in bm25_results)
        
        # 3. Test RRF Hybrid Search
        hybrid_results = retriever.retrieve("cash balance 1000", top_k_semantic=2, top_k_bm25=2)
        assert len(hybrid_results) > 0
        
        # 4. Test Reranking
        config_manager.config["retrieval"]["rerank_top_k"] = 2
        reranker = CrossEncoderReranker(config_manager=config_manager)
        reranked = reranker.rerank("When does payout of funds complete?", hybrid_results, top_k=2)
        assert len(reranked) > 0
        assert "rerank_score" in reranked[0]
