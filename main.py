import argparse
import sys
from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.transformer import QueryTransformer
from src.generation.generator import GroundedGenerator

def run_local_query(query: str):
    """Executes a single RAG query in-process and prints the result to console."""
    config_manager = ConfigManager()
    
    # Initialize classes
    print("Initializing RAG pipeline components...")
    transformer = QueryTransformer(config_manager)
    retriever = HybridRetriever(config_manager)
    reranker = CrossEncoderReranker(config_manager)
    generator = GroundedGenerator(config_manager)
    
    print(f"User Query: '{query}'")
    
    # 1. Transform query
    print("Expanding query...")
    sub_queries = transformer.transform(query)
    
    # 2. Retrieve candidates
    print("Retrieving candidates...")
    all_candidates = []
    seen_ids = set()
    for sq in sub_queries:
        candidates = retriever.retrieve(sq)
        for cand in candidates:
            if cand["id"] not in seen_ids:
                seen_ids.add(cand["id"])
                all_candidates.append(cand)
                
    # 3. Rerank candidates
    print(f"Retrieved {len(all_candidates)} candidates. Reranking...")
    reranked = reranker.rerank(query, all_candidates)
    
    # 4. Generate grounded answer
    print("Generating grounded answer...")
    answer = generator.generate(query, reranked)
    
    # Print results formatted nicely
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer.answer)
    print("\n" + "="*50)
    print("CITATIONS:")
    print("="*50)
    if answer.citations:
        for idx, cit in enumerate(answer.citations):
            print(f"[{idx+1}] Source: {cit.source} | {cit.clause_id} ({cit.clause_title})")
            print(f"    Snippet: \"{cit.snippet}\"")
    else:
        print("No citations generated.")
        
    print("\n" + "="*50)
    print("RULES & timelines IDENTIFIED:")
    print("="*50)
    print(f"Applicable Rules: {answer.applicable_rules}")
    print(f"Thresholds & Timelines: {answer.thresholds_and_timelines}")
    print(f"Required Actions: {answer.required_actions}")
    print(f"Grounding Confidence: {answer.grounding_confidence:.2f}")
    print(f"Is Sufficient: {answer.is_sufficient}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="Brokerage Rules & Trading Policy Assistant — Core CLI Driver"
    )
    
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the document ingestion pipeline to clean, chunk, and index synthetic policy documents."
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the FastAPI backend server on host and port configured in global_config.yaml."
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Execute a single natural-language query programmatically against the local RAG pipeline."
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run RAGAS evaluation and LLM performance comparison on the golden dataset."
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executes ingestion first, then launches the FastAPI backend server."
    )
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    if args.ingest or args.all:
        print("Starting ingestion pipeline...")
        config_manager = ConfigManager()
        pipeline = IngestionPipeline(config_manager)
        pipeline.run()
        print("Ingestion completed successfully.")
        
    if args.api or args.all:
        from src.interface.api import start_server
        start_server()
        
    if args.query:
        run_local_query(args.query)
        
    if args.eval:
        print("Running pipeline evaluation comparison...")
        from src.evaluation.harness import RagasEvaluator
        evaluator = RagasEvaluator(config_manager=ConfigManager())
        evaluator.run_comparison()
        print("Evaluation comparison run completed. Reports committed to docs/ folder.")

if __name__ == "__main__":
    main()
