"""
This module contains the Deduplicator class, which is used to find and mark duplicate
documents in a corpus.

Uses MinHash + LSH for approximate matching, with configurable candidate pair limits
to control memory usage on resource-constrained systems.
"""

import logging
from typing import Dict, List, Optional, Set

import networkx as nx
from datasketch import MinHash, MinHashLSH

from cubo.config import config

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Finds and marks duplicate documents in a corpus using MinHash and LSH.

    Features:
    - Configurable similarity threshold and MinHash permutations
    - Candidate pair cap to limit memory on constrained systems
    - Uses config.deduplication.max_candidates for laptop mode
    """

    def __init__(
        self, threshold: float = 0.8, num_perm: int = 128, max_candidates: Optional[int] = None
    ):
        """Initialize the Deduplicator.

        Args:
            threshold: Jaccard similarity threshold for duplicates (0-1)
            num_perm: Number of MinHash permutations (higher = more accurate)
            max_candidates: Maximum candidate pairs to consider (None = unlimited)
        """
        self.threshold = threshold
        self.num_perm = num_perm

        # Get max_candidates from config if not specified
        if max_candidates is None:
            dedup_config = config.get("deduplication", {})
            if isinstance(dedup_config, dict):
                max_candidates = dedup_config.get("max_candidates")
        self.max_candidates = max_candidates

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # Statistics
        self._total_pairs_found = 0
        self._pairs_used = 0

    def deduplicate(self, documents: List[Dict]) -> Dict[str, str]:
        """
        Finds duplicate documents and returns a mapping from each document to its
        canonical version.

        :param documents: A list of dictionaries, where each dictionary represents a
                          document and has at least a 'doc_id' and 'text' key.
        :return: A dictionary mapping each document ID to its canonical document ID.
        """
        minhashes = self._create_minhashes(documents)
        self._index_minhashes(minhashes)
        candidate_pairs = self._find_candidate_pairs(minhashes)
        graph = self._build_graph(candidate_pairs)
        clusters = self._find_clusters(graph)
        canonical_map = self._get_canonical_map(clusters, documents)
        return canonical_map

    def _create_minhashes(self, documents: List[Dict]) -> Dict[str, MinHash]:
        """Creates MinHash objects for a list of documents."""
        minhashes = {}
        for doc in documents:
            minhash = MinHash(num_perm=self.num_perm)
            for word in set(doc["text"].split()):
                minhash.update(word.encode("utf8"))
            minhashes[doc["doc_id"]] = minhash
        return minhashes

    def _index_minhashes(self, minhashes: Dict[str, MinHash]):
        """Indexes the MinHash objects in the LSH index."""
        for doc_id, minhash in minhashes.items():
            self.lsh.insert(doc_id, minhash)

    def _find_candidate_pairs(self, minhashes: Dict[str, MinHash]) -> Set[tuple]:
        """Finds candidate pairs of duplicates using LSH.

        When max_candidates is set, stops after finding that many pairs
        to limit memory usage. Uses a priority heuristic to process
        documents with more potential matches first.
        """
        candidate_pairs: Set[tuple] = set()

        # If we have a cap, we want to prioritize docs with more matches
        if self.max_candidates:
            # Quick pass to count matches per doc
            match_counts = {}
            for doc_id in minhashes:
                match_counts[doc_id] = len(self.lsh.query(minhashes[doc_id]))

            # Process in order of most matches (more likely to find clusters)
            sorted_docs = sorted(match_counts.keys(), key=lambda x: match_counts[x], reverse=True)

            for doc_id in sorted_docs:
                if len(candidate_pairs) >= self.max_candidates:
                    break

                result = self.lsh.query(minhashes[doc_id])
                for other_doc_id in result:
                    if doc_id != other_doc_id:
                        pair = tuple(sorted((doc_id, other_doc_id)))
                        if pair not in candidate_pairs:
                            candidate_pairs.add(pair)
                            if len(candidate_pairs) >= self.max_candidates:
                                break
        else:
            # Unlimited mode - original behavior
            for doc_id in minhashes:
                result = self.lsh.query(minhashes[doc_id])
                for other_doc_id in result:
                    if doc_id != other_doc_id:
                        candidate_pairs.add(tuple(sorted((doc_id, other_doc_id))))

        # Record stats
        self._total_pairs_found = (
            len(candidate_pairs) if not self.max_candidates else len(candidate_pairs)
        )
        self._pairs_used = len(candidate_pairs)

        if self.max_candidates and len(candidate_pairs) >= self.max_candidates:
            logger.debug(
                f"Deduplication capped at {self.max_candidates} candidate pairs "
                f"(may have missed some duplicates)"
            )

        return candidate_pairs

    def _build_graph(self, candidate_pairs: Set[tuple]) -> nx.Graph:
        """Builds a graph of the candidate pairs."""
        graph = nx.Graph()
        graph.add_edges_from(candidate_pairs)
        return graph

    def _find_clusters(self, graph: nx.Graph) -> List[Set[str]]:
        """Finds the clusters of duplicates."""
        return [cluster for cluster in nx.connected_components(graph)]

    def _get_canonical_map(self, clusters: List[Set[str]], documents: List[Dict]) -> Dict[str, str]:
        """
        Gets the canonical document for each cluster and returns the deduplication map.
        """
        doc_map = {doc["doc_id"]: doc for doc in documents}
        canonical_map = {}
        for cluster in clusters:
            # Choose the longest document as the canonical one
            canonical_doc_id = max(cluster, key=lambda doc_id: len(doc_map[doc_id]["text"]))
            for doc_id in cluster:
                canonical_map[doc_id] = canonical_doc_id

        # Add single documents to the map
        all_docs_in_clusters = set.union(*clusters) if clusters else set()
        for doc in documents:
            if doc["doc_id"] not in all_docs_in_clusters:
                canonical_map[doc["doc_id"]] = doc["doc_id"]

        return canonical_map

    def get_stats(self) -> Dict[str, int]:
        """Return deduplication statistics from the last run.

        Returns:
            Dictionary with stats:
            - pairs_found: Number of candidate pairs discovered
            - pairs_used: Number of pairs actually used (may be capped)
            - max_candidates: The configured cap (None if unlimited)
            - was_capped: Whether the cap was hit
        """
        return {
            "pairs_found": self._total_pairs_found,
            "pairs_used": self._pairs_used,
            "max_candidates": self.max_candidates,
            "was_capped": (
                self.max_candidates is not None and self._pairs_used >= self.max_candidates
            ),
        }
