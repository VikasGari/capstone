from pathlib import Path

def write_evaluation_reports(
    dest_dir: Path,
    primary_model: str,
    fallback_model: str,
    flash_metrics: dict,
    flash_failures: dict,
    pro_metrics: dict,
    pro_failures: dict
) -> None:
    """
    Generates and saves the unified RAGAS metrics and model comparison evaluation report.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    flash_total = sum(flash_failures.values()) or 21
    pro_total = sum(pro_failures.values()) or 21
    
    flash_pct = (flash_failures['successful_runs'] / flash_total) * 100
    pro_pct = (pro_failures['successful_runs'] / pro_total) * 100
    
    # Write unified docs/eval_report.md
    eval_path = dest_path / "eval_report.md"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"""# RAGAS Evaluation & Model Comparison Report

## 1. Metric Summary
| Metric | {primary_model} | {fallback_model} |
|---|---|---|
| Faithfulness | {flash_metrics['faithfulness']:.3f} | {pro_metrics['faithfulness']:.3f} |
| Answer Relevancy | {flash_metrics['answer_relevancy']:.3f} | {pro_metrics['answer_relevancy']:.3f} |
| Context Recall | {flash_metrics['context_recall']:.3f} | {pro_metrics['context_recall']:.3f} |
| Context Precision | {flash_metrics['context_precision']:.3f} | {pro_metrics['context_precision']:.3f} |

## 2. Model Comparison
- **{primary_model}:** Faithfulness={flash_metrics['faithfulness']:.2f}, Answer Relevancy={flash_metrics['answer_relevancy']:.2f}, Success={flash_failures['successful_runs']}/{flash_total} ({flash_pct:.1f}%)
- **{fallback_model}:** Faithfulness={pro_metrics['faithfulness']:.2f}, Answer Relevancy={pro_metrics['answer_relevancy']:.2f}, Success={pro_failures['successful_runs']}/{pro_total} ({pro_pct:.1f}%)

## 3. Failure Analysis
- **{primary_model} Success:** {flash_failures['successful_runs']} / {flash_total} ({flash_pct:.1f}%)
- **{fallback_model} Success:** {pro_failures['successful_runs']} / {pro_total} ({pro_pct:.1f}%)
""")

    print(f"Comparison run completed. Unified report written under {dest_path}/eval_report.md.")
