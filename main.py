"""
RDE-LMC Main Entry Point.

Usage:
    python main.py --mode build_kb
    python main.py --mode single --query "某企业..."
    python main.py --mode evaluate [--limit N] [--no-resume]

Follows Readme pattern: load_kb() -> run_pipeline() -> evaluate()
"""
import os
import sys
import argparse
import json
import yaml

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger

logger = get_logger("main")


def load_kb(config: dict) -> dict:
    """Load the Knowledge Base. Step 1 of Readme pipeline."""
    from src.kb.build_kb import load_kb as _load_kb
    logger.info("Loading Knowledge Base...")
    return _load_kb(config)


def run_pipeline(query: str, config: dict, kb: dict) -> dict:
    """
    Run the full RDE-LMC pipeline for a single query.
    Step 2 of Readme pipeline: retrieval -> discrimination -> execution -> dynamic loop.

    Returns:
        {"numerical_answer": [...], "article_answer": [...]}
    """
    from src.pipeline.rde_pipeline import RDEPipeline
    pipeline = RDEPipeline(config, kb)
    y, A = pipeline.run_case(query)
    return {"numerical_answer": y, "article_answer": A}


def evaluate(config: dict, kb: dict, limit: int = None, resume: bool = True, target_ids: list = None) -> dict:
    """Run evaluation on test dataset. Step 3 of Readme pipeline."""
    from src.evaluation.evaluator import run_evaluation
    return run_evaluation(config=config, kb=kb, limit=limit, resume=resume, target_ids=target_ids)


def main():
    parser = argparse.ArgumentParser(
        description="RDE-LMC: Retrieval, Discrimination, Execution for Legal Mathematical Calculation"
    )
    parser.add_argument(
        "--mode",
        choices=["build_kb", "single", "evaluate"],
        default="evaluate",
        help="Operation mode: build_kb | single | evaluate"
    )
    parser.add_argument("--query", type=str, help="Query for single mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated list of sample IDs to evaluate")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(_project_root, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.mode == "build_kb":
        # Step 1: Build KB
        logger.info("Building Knowledge Base...")
        from src.kb.build_kb import build_kb
        build_kb(config)
        logger.info("KB build complete!")
        return

    # Load KB (needed for single and evaluate modes)
    kb = load_kb(config)

    if args.mode == "single":
        if not args.query:
            print("Error: --query is required for single mode")
            sys.exit(1)

        # Run pipeline
        logger.info(f"Query: {args.query}")
        result = run_pipeline(args.query, config, kb)

        print("\n=== Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == "evaluate":
        # Parse target IDs if provided
        target_ids = None
        if args.ids:
            target_ids = [int(i.strip()) for i in args.ids.split(",")]

        # Run evaluation
        metrics = evaluate(
            config=config,
            kb=kb,
            limit=args.limit,
            resume=not args.no_resume,
            target_ids=target_ids
        )

        print("\n=== Evaluation Metrics ===")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
