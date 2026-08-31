import difflib
import time
import numpy as np
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config.config_manager import ConfigManager

class RagasMetricScore(BaseModel):
    faithfulness: float = Field(description="Score between 0.0 and 1.0 of grounding (1.0 = fully grounded, 0.0 = hallucinated).")
    answer_relevancy: float = Field(description="Score between 0.0 and 1.0 of relevance to user query.")
    context_recall: float = Field(description="Score between 0.0 and 1.0 of ground truth facts recall.")
    context_precision: float = Field(description="Score between 0.0 and 1.0 of precision of retrieved contexts.")

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial RAG evaluation judge."),
    ("human", """Rate these RAGAS metrics (scores 0.0 to 1.0) based strictly on context details:
1. Faithfulness (answer grounded in Contexts)
2. Answer Relevancy (addresses Question directly)
3. Context Recall (Contexts contain Ground Truth facts)
4. Context Precision (retrieved Contexts are highly relevant)

Q: "{question}"
A: "{answer}"
GT: "{ground_truth}"
Contexts:
{context_block}
""")
])

class RagasJudge:
    """
    Evaluates individual QA outputs against ground truth references and context blocks.
    Computes RAGAS metrics via LangChain LCEL judge chains with deterministic fallbacks.
    Respects rate limiting settings configured in config.yaml.
    """
    def __init__(self, judge_model: str, api_key: str = None, config_manager: ConfigManager = None):
        self.judge_model = judge_model
        self.api_key = api_key
        self.config_manager = config_manager or ConfigManager()
        self.judge_chain = None
        
        gen_cfg = self.config_manager.get_section("generation")
        self.rate_limit_delay = gen_cfg.get("rate_limit_delay_seconds")
        
        if self.api_key:
            try:
                judge_llm = ChatGoogleGenerativeAI(
                    model=self.judge_model,
                    temperature=0.0,
                    google_api_key=self.api_key
                ).with_structured_output(RagasMetricScore)
                self.judge_chain = JUDGE_PROMPT | judge_llm
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain judge chain: {e}")

    def judge_single(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> RagasMetricScore:
        """Rates RAGAS metrics for a single query result."""
        if not self.judge_chain:
            return self._fallback_metrics(question, answer, contexts, ground_truth)

        context_block = "\n\n".join([f"Context [{i+1}]: {c}" for i, c in enumerate(contexts)])
        payload = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "context_block": context_block
        }
        try:
            return self.judge_chain.invoke(payload)
        except Exception:
            return self._fallback_metrics(question, answer, contexts, ground_truth)

    def _fallback_metrics(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> RagasMetricScore:
        """Deterministic string similarity fallback metric calculation."""
        ans_clean = answer.lower()
        gt_clean = ground_truth.lower()
        ctx_clean = " ".join(contexts).lower()
        
        faith = difflib.SequenceMatcher(None, ans_clean, ctx_clean).ratio() if ctx_clean else 1.0
        rel = difflib.SequenceMatcher(None, ans_clean, gt_clean).ratio()
        rec = difflib.SequenceMatcher(None, gt_clean, ctx_clean).ratio() if ctx_clean else 0.0
        prec = 1.0 if ctx_clean else 0.0
        return RagasMetricScore(faithfulness=faith, answer_relevancy=rel, context_recall=rec, context_precision=prec)

    def evaluate_results(self, pipeline_results: list[dict]) -> dict:
        """Aggregates RAGAS metrics over a list of QA outputs."""
        faith_scores, rel_scores, rec_scores, prec_scores = [], [], [], []
        for idx, r in enumerate(pipeline_results):
            score = self.judge_single(r["question"], r["answer"], r["contexts"], r["ground_truth"])
            faith_scores.append(score.faithfulness)
            rel_scores.append(score.answer_relevancy)
            rec_scores.append(score.context_recall)
            prec_scores.append(score.context_precision)
            print(f"[{idx+1}/{len(pipeline_results)}] Rated: F={score.faithfulness:.2f} | R={score.answer_relevancy:.2f}", flush=True)
            
            # Rate limiting guardrail
            if self.rate_limit_delay is not None and float(self.rate_limit_delay) > 0:
                time.sleep(float(self.rate_limit_delay))
                
        return {
            "faithfulness": float(np.mean(faith_scores)),
            "answer_relevancy": float(np.mean(rel_scores)),
            "context_recall": float(np.mean(rec_scores)),
            "context_precision": float(np.mean(prec_scores))
        }

    def analyze_failures(self, results: list[dict]) -> dict:
        """Classifies QA outputs into failure taxonomy categories."""
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
