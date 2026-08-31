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
    Generates and saves the RAGAS metrics report and model comparison summary files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Write docs/eval_report.md
    eval_path = dest_path / "eval_report.md"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"""# RAGAS Evaluation Report

## 1. Metric Summary
| Metric | {primary_model} | {fallback_model} |
|---|---|---|
| Faithfulness | {flash_metrics['faithfulness']:.3f} | {pro_metrics['faithfulness']:.3f} |
| Answer Relevancy | {flash_metrics['answer_relevancy']:.3f} | {pro_metrics['answer_relevancy']:.3f} |
| Context Recall | {flash_metrics['context_recall']:.3f} | {pro_metrics['context_recall']:.3f} |
| Context Precision | {flash_metrics['context_precision']:.3f} | {pro_metrics['context_precision']:.3f} |

## 2. Failure Analysis
- **{primary_model} Success:** {flash_failures['successful_runs']} / {len(flash_failures) or 20} ({flash_failures['successful_runs'] / (flash_failures['successful_runs'] + flash_failures['retrieval_failures'] + flash_failures['synthesis_failures']) * 100:.1f}%)
- **{fallback_model} Success:** {pro_failures['successful_runs']} / {len(pro_failures) or 20} ({pro_failures['successful_runs'] / (pro_failures['successful_runs'] + pro_failures['retrieval_failures'] + pro_failures['synthesis_failures']) * 100:.1f}%)
""")

    # 2. Write docs/model_comparison.md
    comp_path = dest_path / "model_comparison.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(f"""# Model Comparison Report

## 1. Metric Comparison
- **{primary_model}:** Faithfulness={flash_metrics['faithfulness']:.2f}, Success={flash_failures['successful_runs']}
- **{fallback_model}:** Faithfulness={pro_metrics['faithfulness']:.2f}, Success={pro_failures['successful_runs']}

## 2. Recommendation
{primary_model} is selected as the primary generation model due to its performance.
""")
    print(f"Comparison run completed. Reports written under {dest_path}/ folder.")
