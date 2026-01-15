import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer

DATA_DIR = Path(r"C:\Users\paolo\Desktop\cubo\data\italian_legal")
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"

class ItalianTokenizer(MultilingualTokenizer):
    """Force Italian language detection for ablation consistency."""
    def detect_language(self, text: str) -> str:
        return "it"

def load_queries():
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def run_ablation():
    """
    Run ablation study on Italian legal queries to validate the impact
    of Snowball stemming on retrieval recall.
    
    Target Claim: ~4.2% improvement in Recall@10.
    """
    print("--- Italian Legal Query Ablation (Snowball Stemming) ---")
    
    if not CORPUS_PATH.exists() or not QUERIES_PATH.exists():
        print("Data not found. Please run 'tools/generate_italian_data.py' first.")
        return

    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(d)
    
    queries = load_queries()
    print(f"Loaded {len(docs)} documents and {len(queries)} queries.")

    # 1. STEMMING (Baseline)
    print("\n[1] Running with Snowball Stemming (Italian)...")
    searcher_stem = BM25Searcher(backend="python")
    # Patch store to use our forced Italian tokenizer with stemming
    searcher_stem._store.tokenizer = ItalianTokenizer(use_stemming=True)
    # Patch wrapper for consistency
    searcher_stem.tokenizer = searcher_stem._store.tokenizer
    
    searcher_stem.index_documents(docs)
    
    hits_stemming = 0
    for q in queries:
        results = searcher_stem.search(q['text'], top_k=10)
        retrieved_ids = [r['doc_id'] for r in results]
        
        if any(rel in retrieved_ids for rel in q['relevant_docs']):
            hits_stemming += 1
            
    recall_stemming = hits_stemming / len(queries)
    print(f"Recall@10 (Stemming): {recall_stemming:.4f} ({hits_stemming}/{len(queries)})")

    # 2. NO STEMMING (Ablation)
    print("\n[2] Running WITHOUT Stemming...")
    searcher_no_stem = BM25Searcher(backend="python")
    # Patch store to use our forced Italian tokenizer WITHOUT stemming
    searcher_no_stem._store.tokenizer = ItalianTokenizer(use_stemming=False)
    searcher_no_stem.tokenizer = searcher_no_stem._store.tokenizer
    
    searcher_no_stem.index_documents(docs)
    
    hits_no_stemming = 0
    for q in queries:
        results = searcher_no_stem.search(q['text'], top_k=10)
        retrieved_ids = [r['doc_id'] for r in results]
        
        if any(rel in retrieved_ids for rel in q['relevant_docs']):
            hits_no_stemming += 1
            
    recall_no_stemming = hits_no_stemming / len(queries)
    print(f"Recall@10 (No Stemming): {recall_no_stemming:.4f} ({hits_no_stemming}/{len(queries)})")

    # 3. Compare
    delta = recall_stemming - recall_no_stemming
    print("\n--- Results ---")
    print(f"Recall@10 Improvement: +{delta:.4f} points")
    
    if recall_stemming > recall_no_stemming:
        print("SUCCESS: Validated positive impact of stemming on Italian queries.")
        print("Anecdotal evidence supported (e.g., 'gatto'/'gatti', 'contratto'/'contratti').")
    else:
        print("WARNING: No improvement observed.")

if __name__ == "__main__":
    run_ablation()
