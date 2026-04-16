"""
Evaluator for the RDE-LMC system.
Runs the pipeline on the test dataset and computes metrics.
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import get_logger
from src.evaluation.metrics import compute_all_metrics

logger = get_logger(__name__)


def _load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset."""
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _load_existing_results(results_path: str) -> Dict[int, Dict]:
    """Load existing results for resume support."""
    if not os.path.exists(results_path):
        return {}

    existing = {}
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    existing[record["id"]] = record
                except Exception:
                    pass

    logger.info(f"Loaded {len(existing)} existing results")
    return existing


def run_evaluation(
    config: dict,
    kb: dict,
    pipeline=None,
    limit: int = None,
    resume: bool = True,
    target_ids: list = None
) -> Dict:
    """
    Run evaluation on the test dataset.

    Args:
        config: Configuration dict
        kb: Knowledge base dict
        pipeline: Optional pre-initialized pipeline (avoids reloading)
        limit: Only evaluate first N samples
        resume: Skip samples already in results file
        target_ids: If provided, only evaluate samples with these IDs

    Returns:
        Metrics dict
    """
    from src.dynamic.controller import run_dynamic_loop

    project_root = _project_root
    paths = config.get("paths", {})

    dataset_path = os.path.join(project_root, paths.get("dataset_path", "data/dataset/dataset.jsonl"))
    results_path = os.path.join(project_root, paths.get("results_path", "data/results.json"))

    # Load dataset
    samples = _load_dataset(dataset_path)
    if limit is not None:
        samples = samples[:limit]
    if target_ids is not None:
        target_set = set(target_ids)
        samples = [s for s in samples if s["id"] in target_set]

    logger.info(f"Evaluating on {len(samples)} samples")

    # Load existing results for resume
    existing_results = {}
    if resume:
        existing_results = _load_existing_results(results_path)

    # Run pipeline
    predictions = []
    ground_truths = []

    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc="Evaluating")
    except ImportError:
        iterator = samples

    with open(results_path, 'a' if (resume and existing_results) else 'w', encoding='utf-8') as out_f:
        for sample in iterator:
            sample_id = sample["id"]
            query = sample["query"]
            truth_num = sample.get("numerical_answer", [])
            truth_art = sample.get("article_answer", [])

            ground_truths.append({
                "numerical_answer": truth_num,
                "article_answer": truth_art
            })

            # Skip if already computed
            if resume and sample_id in existing_results:
                existing = existing_results[sample_id]
                predictions.append({
                    "numerical_answer": existing.get("predicted_numerical", []),
                    "article_answer": existing.get("predicted_articles", []),
                })
                continue

            # Run pipeline
            try:
                result = run_dynamic_loop(
                    initial_query=query,
                    kb=kb,
                    config=config
                )
                pred_num = result.get("numerical_answer", [])
                pred_art = result.get("article_answer", [])
                success = result.get("success", False)
            except Exception as e:
                logger.error(f"Pipeline failed for sample {sample_id}: {e}")
                pred_num = []
                pred_art = []
                success = False

            predictions.append({
                "numerical_answer": pred_num,
                "article_answer": pred_art,
            })

            # Write result to file
            record = {
                "id": sample_id,
                "query": query,
                "predicted_numerical": pred_num,
                "predicted_articles": pred_art,
                "true_numerical": truth_num,
                "true_articles": truth_art,
                "success": success,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    # Compute metrics
    tolerance = config.get("evaluation", {}).get("numerical_tolerance", 0.01)
    metrics = compute_all_metrics(predictions, ground_truths, tolerance)

    logger.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f} ({v*100:.1f}%)")
        else:
            logger.info(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate RDE-LMC Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    args = parser.parse_args()

    config_path = os.path.join(_project_root, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from src.kb.build_kb import load_kb
    kb = load_kb(config)

    metrics = run_evaluation(config, kb, limit=args.limit, resume=not args.no_resume)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
