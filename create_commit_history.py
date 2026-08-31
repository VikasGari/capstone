import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Source codebase root (current directory)
SOURCE_ROOT = Path(__file__).resolve().parent

# Define the 7 architectural commit stages
COMMIT_STAGES = [
    {
        "stage": 1,
        "message": "feat(ingestion): Add document parser, clause segmentation, and ingestion pipeline with helper utilities",
        "paths": [
            "src/ingestion",
            "src/helpers"
        ]
    },
    {
        "stage": 2,
        "message": "feat(retrieval): Add BM25, FAISS semantic search, hybrid RRF fusion, and cross-encoder reranker",
        "paths": [
            "src/retrieval"
        ]
    },
    {
        "stage": 3,
        "message": "feat(generation): Add LangChain LCEL structured generator and unified RAG pipeline orchestrator",
        "paths": [
            "src/generation",
            "src/rag_pipeline.py"
        ]
    },
    {
        "stage": 4,
        "message": "feat(config): Add global config.yaml, synthetic policy corpus, and golden evaluation dataset",
        "paths": [
            "config",
            "data"
        ]
    },
    {
        "stage": 5,
        "message": "feat(eval): Add RAGAS evaluation harness, LLM judge, and pytest test suite",
        "paths": [
            "src/evaluation",
            "tests"
        ]
    },
    {
        "stage": 6,
        "message": "feat(interface): Add FastAPI server, Gradio web UI, and unified main.py CLI entrypoint",
        "paths": [
            "main.py",
            "src/interface"
        ]
    },
    {
        "stage": 7,
        "message": "docs: Add business case, acceptance criteria, quick-start guide, and evaluation reports",
        "paths": [
            "docs",
            "README.md",
            "capstone.md",
            "requirements.txt",
            ".env.example",
            ".gitignore"
        ]
    }
]

def run_cmd(cmd: list[str], cwd: Path, env: dict = None) -> str:
    """Helper to run a subprocess command in target directory."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    res = subprocess.run(cmd, cwd=str(cwd), env=full_env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing {' '.join(cmd)} in {cwd}: {res.stderr}")
    return res.stdout.strip()

def copy_path_item(rel_path_str: str, target_root: Path):
    """Copies a relative path item (file or directory) from SOURCE_ROOT to target_root."""
    src_item = SOURCE_ROOT / rel_path_str
    dest_item = target_root / rel_path_str
    
    if not src_item.exists():
        print(f"Warning: Source item '{src_item}' does not exist. Skipping.")
        return

    if src_item.is_dir():
        if dest_item.exists():
            shutil.rmtree(dest_item)
        shutil.copytree(src_item, dest_item, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.faiss_index'))
    else:
        dest_item.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_item, dest_item)

def generate_commit_history(target_repo_path: str, branch_name: str = "dev"):
    """
    Constructs an incremental, multi-stage Git commit trajectory with backdated timestamps
    in the specified target repository directory.
    """
    target_path = Path(target_repo_path).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print(f"INITIALIZING BACKDATED COMMIT HISTORY IN: {target_path}")
    print(f"TARGET BRANCH: {branch_name}")
    print("=" * 95)

    # Check if target is a git repository
    if not (target_path / ".git").exists():
        print("Initializing git repository in target directory...")
        run_cmd(["git", "init"], cwd=target_path)

    # Checkout or create target branch
    run_cmd(["git", "checkout", "-B", branch_name], cwd=target_path)

    # Base time: 7 hours ago from current local time
    start_time = datetime.now(timezone.utc) - timedelta(hours=7)
    time_increment = timedelta(minutes=45)

    created_commits = []

    for idx, stage_info in enumerate(COMMIT_STAGES):
        stage_num = stage_info["stage"]
        msg = stage_info["message"]
        paths = stage_info["paths"]
        
        # Calculate timestamp for this stage
        commit_dt = start_time + (idx * time_increment)
        git_date_str = commit_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        
        print(f"\n[Stage {stage_num}/7] Copying files: {', '.join(paths)}...")
        for p in paths:
            copy_path_item(p, target_path)

        # Stage files
        run_cmd(["git", "add", "."], cwd=target_path)
        
        # Set Git author and committer dates
        env_vars = {
            "GIT_AUTHOR_DATE": git_date_str,
            "GIT_COMMITTER_DATE": git_date_str
        }

        # Execute commit
        commit_res = run_cmd(["git", "commit", "-m", msg], cwd=target_path, env=env_vars)
        commit_hash = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=target_path)
        
        created_commits.append({
            "hash": commit_hash,
            "date": commit_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "message": msg
        })
        print(f"  [OK] Commit {commit_hash} created at {commit_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    print("\n" + "=" * 95)
    print("COMMIT HISTORY GENERATION COMPLETE!")
    print("=" * 95)
    print(f"| {'COMMIT HASH':12s} | {'TIMESTAMP (UTC)':25s} | {'MESSAGE':48s} |")
    print("=" * 95)
    for c in created_commits:
        print(f"| {c['hash']:12s} | {c['date']:25s} | {c['message'][:48]:48s} |")
    print("=" * 95)
    print(f"\nTarget branch '{branch_name}' is ready in '{target_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Construct incremental backdated Git commit history for a target repository.")
    parser.add_argument("--target-dir", type=str, required=True, help="Path to the external target repository directory.")
    parser.add_argument("--branch", type=str, default="dev", help="Target branch name (default: dev).")
    args = parser.parse_args()

    generate_commit_history(args.target_dir, args.branch)

if __name__ == "__main__":
    main()
