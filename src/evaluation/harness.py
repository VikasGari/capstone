import json
from pathlib import Path
from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.evaluation.judge import RagasJudge, RagasMetricScore
from src.helpers import write_evaluation_reports

class RagasEvaluator:
    """
    Evaluation harness for the Brokerage Assistant RAG pipeline.
    Runs pipeline over data/golden_set.json, rates metrics using LangChain LCEL judge chain,
    and writes simple comparative reports under docs/.
    """
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.golden_set_path = Path(self.config_manager.get_section("paths")["golden_set_path"])
        self.api_key = self.config_manager.get_env_var("GEMINI_API_KEY")

        # Load models dynamically from configuration settings
        gen_cfg = self.config_manager.get_section("generation")
        self.primary_model = gen_cfg.get("primary_model", "gemini-3.5-flash-lite")
        self.fallback_model = gen_cfg.get("fallback_model", "gemini-2.5-flash-lite")

        # Instantiate unified RAG execution orchestrator and Judge
        self.rag_pipeline = RAGPipeline(self.config_manager)
        self.judge = RagasJudge(self.primary_model, self.api_key)

    def load_golden_set(self) -> list[dict]:
        """Loads the golden evaluation set."""
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set not found at: {self.golden_set_path}")
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_pipeline_on_dataset(self, model_name: str) -> list[dict]:
        """Executes the RAG pipeline for all questions in the golden dataset."""
        golden_set = self.load_golden_set()
        print(f"Running pipeline with '{model_name}' on {len(golden_set)} queries...")
        
        results = []
        for idx, entry in enumerate(golden_set):
            question = entry["question"]
            ground_truth = entry["reference"]
            ac_id = entry.get("ac_id", "AC-02")
            
            # Call unified RAG execution pathway
            ans_obj, reranked = self.rag_pipeline.run_query(question, model_name=model_name)
            
            results.append({
                "question": question,
                "answer": ans_obj.answer,
                "contexts": [c["document"] for c in reranked],
                "ground_truth": ground_truth,
                "is_sufficient": ans_obj.is_sufficient,
                "ac_id": ac_id,
                "confidence": ans_obj.grounding_confidence
            })
            print(f"[{idx+1}/{len(golden_set)}] Processed query.")
        return results

    def run_comparison(self):
        """Runs evaluation comparisons on candidate models and commits clean, simple reports."""
        # Ensure collection is loaded
        print("Checking database index...")
        try:
            self.rag_pipeline.retriever._initialize_bm25()
        except Exception:
            pipeline = IngestionPipeline(self.config_manager)
            pipeline.run()
            self.rag_pipeline.retriever._initialize_bm25()

        flash_results = self.run_pipeline_on_dataset(self.primary_model)
        flash_metrics = self.judge.evaluate_results(flash_results)
        flash_failures = self.judge.analyze_failures(flash_results)
        
        pro_results = self.run_pipeline_on_dataset(self.fallback_model)
        pro_metrics = self.judge.evaluate_results(pro_results)
        pro_failures = self.judge.analyze_failures(pro_results)

        write_evaluation_reports(
            Path("docs"),
            self.primary_model,
            self.fallback_model,
            flash_metrics,
            flash_failures,
            pro_metrics,
            pro_failures
        )
