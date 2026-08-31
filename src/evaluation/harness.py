import os
import json
import time
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from config.config_manager import ConfigManager
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.transformer import QueryTransformer
from src.generation import GroundedGenerator, GroundedAnswer
from src.helpers import write_evaluation_reports

class RagasMetricScore(BaseModel):
    faithfulness: float = Field(description="Score between 0.0 and 1.0 of grounding (1.0 = fully grounded, 0.0 = hallucinated).")
    answer_relevancy: float = Field(description="Score between 0.0 and 1.0 of relevance to user query.")
    context_recall: float = Field(description="Score between 0.0 and 1.0 of ground truth facts recall.")
    context_precision: float = Field(description="Score between 0.0 and 1.0 of precision of retrieved contexts.")

class RagasEvaluator:
    """
    Simple evaluation harness for the Brokerage Assistant RAG pipeline.
    Runs pipeline over data/golden_set.json, rates metrics using Gemini judge,
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

        self.transformer = QueryTransformer(self.config_manager)
        self.retriever = HybridRetriever(self.config_manager)
        self.reranker = CrossEncoderReranker(self.config_manager)
        self.generator = GroundedGenerator(self.config_manager)

        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.api_exhausted = False

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
            
            sub_queries = [question] if self.api_exhausted else self.transformer.transform(question)
            
            all_candidates = []
            seen_ids = set()
            for sq in sub_queries:
                for cand in self.retriever.retrieve(sq):
                    if cand["id"] not in seen_ids:
                        seen_ids.add(cand["id"])
                        all_candidates.append(cand)
                        
            reranked = self.reranker.rerank(question, all_candidates)
            ans_obj = self.generator.generate(question, reranked, model_name=model_name)
            
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

    def _judge_with_gemini(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> RagasMetricScore:
        """Uses Gemini to rate the RAGAS metrics for a single QA result."""
        if not self.client or self.api_exhausted:
            return self._fallback_metrics(question, answer, contexts, ground_truth)

        context_block = "\n\n".join([f"Context [{i+1}]: {c}" for i, c in enumerate(contexts)])
        prompt = f"""
Rate these RAGAS metrics (scores 0.0 to 1.0) based strictly on context details:
1. Faithfulness (answer grounded in Contexts)
2. Answer Relevancy (addresses Question directly)
3. Context Recall (Contexts contain Ground Truth facts)
4. Context Precision (retrieved Contexts are highly relevant)

Q: "{question}"
A: "{answer}"
GT: "{ground_truth}"
Contexts:
{context_block}

Return JSON with keys: faithfulness, answer_relevancy, context_recall, context_precision
"""
        try:
            # Use configured primary model for evaluation judges
            response = self.client.models.generate_content(
                model=self.primary_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RagasMetricScore,
                    temperature=0.0
                )
            )
            data = json.loads(response.text.strip())
            return RagasMetricScore(**data)
        except Exception:
            return self._fallback_metrics(question, answer, contexts, ground_truth)

    def _fallback_metrics(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> RagasMetricScore:
        """Simple offline fallback logic."""
        import difflib
        ans_clean = answer.lower()
        gt_clean = ground_truth.lower()
        ctx_clean = " ".join(contexts).lower()
        
        faith = difflib.SequenceMatcher(None, ans_clean, ctx_clean).ratio() if ctx_clean else 1.0
        rel = difflib.SequenceMatcher(None, ans_clean, gt_clean).ratio()
        rec = difflib.SequenceMatcher(None, gt_clean, ctx_clean).ratio() if ctx_clean else 0.0
        prec = 1.0 if ctx_clean else 0.0
        return RagasMetricScore(faithfulness=faith, answer_relevancy=rel, context_recall=rec, context_precision=prec)

    def evaluate_results(self, pipeline_results: list[dict]) -> dict:
        """Runs evaluation over the list of pipeline results."""
        faith_scores, rel_scores, rec_scores, prec_scores = [], [], [], []
        for idx, r in enumerate(pipeline_results):
            score = self._judge_with_gemini(r["question"], r["answer"], r["contexts"], r["ground_truth"])
            faith_scores.append(score.faithfulness)
            rel_scores.append(score.answer_relevancy)
            rec_scores.append(score.context_recall)
            prec_scores.append(score.context_precision)
            print(f"[{idx+1}/{len(pipeline_results)}] Rated: F={score.faithfulness:.2f} | R={score.answer_relevancy:.2f}")
        return {
            "faithfulness": float(np.mean(faith_scores)),
            "answer_relevancy": float(np.mean(rel_scores)),
            "context_recall": float(np.mean(rec_scores)),
            "context_precision": float(np.mean(prec_scores))
        }

    def analyze_failures(self, results: list[dict]) -> dict:
        """Groups results by categories to identify failures."""
        ret_fail, ground_fail, synth_fail, success = 0, 0, 0, 0
        for r in results:
            if not r["is_sufficient"]:
                if r["ac_id"] == "AC-05":
                    success += 1
                else:
                    ret_fail += 1
            else:
                overlap = len([w for w in r["ground_truth"].lower().split() if w in r["answer"].lower()]) / max(len(r["ground_truth"].split()), 1)
                if overlap < 0.2:
                    synth_fail += 1
                else:
                    success += 1
        return {"retrieval_failures": ret_fail, "grounding_failures": ground_fail, "synthesis_failures": synth_fail, "successful_runs": success}

    def run_comparison(self):
        """Runs evaluation comparisons on candidate models and commits clean, simple reports."""
        # Ensure collection is loaded
        print("Checking database index...")
        try:
            self.retriever._initialize_bm25()
        except Exception:
            pipeline = IngestionPipeline(self.config_manager)
            pipeline.run()
            self.retriever._initialize_bm25()

        flash_results = self.run_pipeline_on_dataset(self.primary_model)
        flash_metrics = self.evaluate_results(flash_results)
        flash_failures = self.analyze_failures(flash_results)
        
        pro_results = self.run_pipeline_on_dataset(self.fallback_model)
        pro_metrics = self.evaluate_results(pro_results)
        pro_failures = self.analyze_failures(pro_results)

        write_evaluation_reports(
            Path("docs"),
            self.primary_model,
            self.fallback_model,
            flash_metrics,
            flash_failures,
            pro_metrics,
            pro_failures
        )
