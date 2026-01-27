from cubo.adapters.beir_adapter import CuboBeirAdapter

print("Creating adapter")
adapter = CuboBeirAdapter(index_dir="results/tmp_index_small", lightweight=False)
print("Starting index")
count = adapter.index_corpus("data/beir/nfcorpus/corpus.jsonl", "results/tmp_index_small", limit=2)
print("Indexed", count)
