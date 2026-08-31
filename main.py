import argparse

def main():
    parser = argparse.ArgumentParser(description="Brokerage Rules & Trading Policy Assistant Entrypoint")
    parser.add_argument("--mode", choices=["serve", "eval", "ingest", "query"], default="serve", help="Execution mode (default: serve)")
    parser.add_argument("--eval", action="store_true", help="Shortcut to run RAGAS 2-model evaluation harness")
    parser.add_argument("--ingest", action="store_true", help="Shortcut to run document ingestion only")
    parser.add_argument("--query", type=str, default=None, help="Execute a single natural-language query via CLI")
    
    args = parser.parse_args()
    
    from config.config_manager import ConfigManager
    config_manager = ConfigManager()

    # 1. Ingestion Mode
    if args.ingest or args.mode == "ingest":
        from src.ingestion.pipeline import IngestionPipeline
        print("Running document ingestion pipeline...")
        pipeline = IngestionPipeline(config_manager)
        count = pipeline.run()
        print(f"Ingestion completed. {count} chunks indexed.")
        return

    # 2. Evaluation Mode
    if args.eval or args.mode == "eval":
        from src.evaluation.harness import RagasEvaluator
        print("Running RAGAS evaluation harness & model comparison...")
        evaluator = RagasEvaluator(config_manager)
        evaluator.run_comparison()
        print("Evaluation completed. Reports generated under docs/.")
        return

    # 3. Direct CLI Query Mode
    if args.query or args.mode == "query":
        from src.rag_pipeline import RAGPipeline
        query_text = args.query or "What is the pre-open session order entry window?"
        print(f"Querying Assistant via CLI: '{query_text}'")
        pipeline = RAGPipeline(config_manager)
        answer, _ = pipeline.run_query(query_text)
        print("\n--- GROUNDED ANSWER ---")
        print(answer.answer)
        print("\n--- CITATIONS ---")
        for c in answer.citations:
            print(f"- [{c.source_document} | {c.clause_id} {c.clause_title}] {c.quote}")
        return

    # 4. Default: Ingest & Launch Full Web Service (FastAPI + Gradio)
    from src.ingestion.pipeline import IngestionPipeline
    from src.interface.api import start_server

    print("Running document ingestion pipeline...")
    pipeline = IngestionPipeline(config_manager)
    pipeline.run()
    print("Ingestion pipeline completed.")
    
    print("Launching Unified FastAPI + Gradio Web Server...")
    start_server()

if __name__ == "__main__":
    main()
