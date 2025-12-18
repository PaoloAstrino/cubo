"""CLI entry-point for hybrid deduplication workflows."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from cubo.config import config
from cubo.deduplication.deduplicator import Deduplicator
from cubo.deduplication.semantic_deduplicator import (
    DeduplicationResult,
    HybridDeduplicator,
)
from cubo.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deduplication pipeline.")
    parser.add_argument(
        "--input-parquet", required=True, help="Path to the Parquet file with the documents."
    )
    parser.add_argument("--output-map", required=True, help="Path to save the deduplication map.")
    parser.add_argument(
        "--method",
        choices=["minhash", "semantic", "hybrid"],
        default=config.get("deduplication.method", "hybrid"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for MinHash/prefilter.",
    )
    parser.add_argument(
        "--num-perm", type=int, default=128, help="Number of permutations for MinHash/prefilter."
    )
    parser.add_argument(
        "--embeddings", help="Path to precomputed embeddings (required for semantic/hybrid)."
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=config.get("deduplication.similarity_threshold", 0.92),
    )
    parser.add_argument("--ann-backend", default=config.get("deduplication.ann.backend", "faiss"))
    parser.add_argument("--ann-k", type=int, default=config.get("deduplication.ann.k", 50))
    parser.add_argument(
        "--representative-metric",
        default=config.get("deduplication.representative_metric", "summary_score"),
    )
    parser.add_argument(
        "--disable-prefilter",
        action="store_true",
        help="Skip MinHash prefilter for semantic/hybrid runs.",
    )
    parser.add_argument(
        "--run-on",
        choices=["scaffold", "chunks"],
        default=config.get("deduplication.run_on", "scaffold"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading documents from %s", args.input_parquet)
    df = pd.read_parquet(args.input_parquet)
    if "chunk_id" not in df.columns:
        raise ValueError("Input parquet must contain a 'chunk_id' column")

    if args.method == "minhash":
        result = _run_minhash(df, args)
    else:
        if not args.embeddings:
            raise ValueError("--embeddings is required for semantic/hybrid deduplication")
        result = _run_semantic(df, args)

    _write_output_map(args.output_map, result)
    logger.info(
        "Deduplication complete: %s canonical entries from %s chunks",
        len(set(result.canonical_map.values())),
        len(df),
    )


def _run_minhash(df: pd.DataFrame, args: argparse.Namespace) -> DeduplicationResult:
    documents = df.rename(columns={"chunk_id": "doc_id"}).to_dict("records")
    deduplicator = Deduplicator(threshold=args.threshold, num_perm=args.num_perm)
    canonical_map = deduplicator.deduplicate(documents)

    clusters: List[Set[str]] = []
    representatives: Dict[int, Dict[str, Any]] = {}
    cluster_mapping: Dict[str, int] = {}
    grouped: Dict[str, List[str]] = {}
    for chunk_id, canonical in canonical_map.items():
        grouped.setdefault(canonical, []).append(chunk_id)

    row_lookup = df.set_index("chunk_id", drop=False)
    for canonical, members in grouped.items():
        cid = len(clusters)
        clusters.append(set(members))
        row = row_lookup.loc[canonical]
        score = _metric_from_row(row, args.representative_metric)
        representatives[cid] = {"chunk_id": canonical, "score": score, "cluster_size": len(members)}
        for member in members:
            cluster_mapping[member] = cid

    metadata = {
        "method": "minhash",
        "threshold": args.threshold,
        "num_perm": args.num_perm,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    return DeduplicationResult(canonical_map, cluster_mapping, representatives, clusters, metadata)


def _run_semantic(df: pd.DataFrame, args: argparse.Namespace) -> DeduplicationResult:
    prefilter_cfg = {
        "use_minhash": not args.disable_prefilter,
        "num_perm": args.num_perm,
        "minhash_threshold": args.threshold,
    }
    ann_cfg = {"backend": args.ann_backend, "k": args.ann_k}
    clustering_cfg = config.get("deduplication.clustering", {})
    deduplicator = HybridDeduplicator(
        method=args.method,
        similarity_threshold=args.similarity_threshold,
        representative_metric=args.representative_metric,
        prefilter=prefilter_cfg,
        ann=ann_cfg,
        clustering=clustering_cfg,
        run_on=args.run_on,
    )
    return deduplicator.run(df, args.embeddings)


def _write_output_map(path: str, result: DeduplicationResult) -> None:
    payload = {
        "version": "1.0",
        "created_at": datetime.datetime.utcnow().isoformat(),
        "metadata": result.metadata,
        "canonical_map": result.canonical_map,
        "clusters": {str(i): list(cluster) for i, cluster in enumerate(result.clusters)},
        "representatives": {str(cid): rep for cid, rep in result.representatives.items()},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _metric_from_row(row: pd.Series, metric: str) -> float:  # type: ignore[name-defined]
    if metric == "summary_score" and "summary_score" in row and pd.notna(row["summary_score"]):
        return float(row["summary_score"])
    if "text" in row and isinstance(row["text"], str):
        return float(len(row["text"]))
    return 0.0


if __name__ == "__main__":
    main()
