import json
from pathlib import Path

from cubo.adapters.beir_adapter import CuboBeirAdapter


class StubEmbeddingGenerator:
    def __init__(self, dim=8):
        self._dim = dim
        class M:
            def get_sentence_embedding_dimension(self_inner):
                return dim
        self.model = M()

    def encode(self, texts):
        # deterministic embeddings: length-prefixed vector
        import numpy as np
        out = []
        for t in texts:
            v = [float(len(t) % 7) + i * 0.01 for i in range(self._dim)]
            out.append(v)
        return np.array(out, dtype=float)


def test_beir_adapter_smoke_indexes_and_retrieves(tmp_path):
    # Build a tiny corpus
    corpus = tmp_path / "corpus.jsonl"
    docs = [
        {"_id": 1, "title": "alpha", "text": "one two three"},
        {"_id": 2, "title": "beta", "text": "four five six"},
        {"_id": 3, "title": "gamma", "text": "seven eight nine"},
    ]
    with open(corpus, 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    index_dir = tmp_path / "index"
    emb = StubEmbeddingGenerator(dim=8)
    adapter = CuboBeirAdapter(index_dir=str(index_dir), embedding_generator=emb, lightweight=False)

    counted = adapter.index_corpus(str(corpus), str(index_dir), batch_size=2, limit=None, normalize=False)
    assert counted == 3

    # Load and run a small query
    adapter.load_index(str(index_dir))
    res = adapter.retrieve_bulk({"q1": "alpha beta"}, top_k=2)

    assert "q1" in res
    hits = res["q1"]
    # All keys should be non-null strings
    assert all(isinstance(k, str) and k.startswith("beir_") for k in hits.keys())
    # All values should be floats
    assert all(isinstance(v, float) for v in hits.values())
