from config.config_manager import ConfigManager
from src.evaluation.harness import RagasEvaluator

def main():
    print("Running pipeline evaluation comparison...")
    evaluator = RagasEvaluator(config_manager=ConfigManager())
    evaluator.run_comparison()
    print("Evaluation comparison completed. Reports written to docs/eval_report.md and docs/model_comparison.md.")

if __name__ == "__main__":
    main()
