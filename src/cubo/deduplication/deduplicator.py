"""
This module contains the Deduplicator class, which is used to find and mark duplicate
documents in a corpus.
"""
from typing import List, Dict, Set
from datasketch import MinHash, MinHashLSH
import networkx as nx

class Deduplicator:
    """
    Finds and marks duplicate documents in a corpus using MinHash and LSH.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

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
            for word in set(doc['text'].split()):
                minhash.update(word.encode('utf8'))
            minhashes[doc['doc_id']] = minhash
        return minhashes

    def _index_minhashes(self, minhashes: Dict[str, MinHash]):
        """Indexes the MinHash objects in the LSH index."""
        for doc_id, minhash in minhashes.items():
            self.lsh.insert(doc_id, minhash)

    def _find_candidate_pairs(self, minhashes: Dict[str, MinHash]) -> Set[tuple]:
        """Finds candidate pairs of duplicates using LSH."""
        candidate_pairs = set()
        for doc_id in minhashes:
            result = self.lsh.query(minhashes[doc_id])
            for other_doc_id in result:
                if doc_id != other_doc_id:
                    candidate_pairs.add(tuple(sorted((doc_id, other_doc_id))))
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
        doc_map = {doc['doc_id']: doc for doc in documents}
        canonical_map = {}
        for cluster in clusters:
            # Choose the longest document as the canonical one
            canonical_doc_id = max(cluster, key=lambda doc_id: len(doc_map[doc_id]['text']))
            for doc_id in cluster:
                canonical_map[doc_id] = canonical_doc_id
        
        # Add single documents to the map
        all_docs_in_clusters = set.union(*clusters) if clusters else set()
        for doc in documents:
            if doc['doc_id'] not in all_docs_in_clusters:
                canonical_map[doc['doc_id']] = doc['doc_id']

        return canonical_map
