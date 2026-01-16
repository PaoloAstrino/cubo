import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer

BASE_DATA_DIR = Path(r"C:\Users\paolo\Desktop\cubo\data")

class ForcedLanguageTokenizer(MultilingualTokenizer):
    """Force specific language detection for ablation consistency."""
    def __init__(self, language, use_stemming=True):
        super().__init__(use_stemming=use_stemming)
        self.forced_language = language
        
    def detect_language(self, text: str) -> str:
        return self.forced_language

def run_lang_ablation(lang_code):
    data_dir = BASE_DATA_DIR / f"legal_{lang_code}"
    if lang_code == "it":
        data_dir = BASE_DATA_DIR / "italian_legal"
        
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"

    if not corpus_path.exists() or not queries_path.exists():
        print(f"Data for {lang_code} not found at {data_dir}.")
        return None, None

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

def calculate_metrics(searcher, queries, top_k=10):
    total_recall = 0
    total_precision = 0
    
    for q in queries:
        results = searcher.search(q['text'], top_k=top_k)
        retrieved_ids = [r['doc_id'] for r in results]
        relevant_ids = set(q['relevant_docs'])
        
        true_positives = len([rid for rid in retrieved_ids if rid in relevant_ids])
        
        recall = true_positives / len(relevant_ids) if relevant_ids else 0
        precision = true_positives / top_k
        
        total_recall += recall
        total_precision += precision
        
    avg_recall = total_recall / len(queries)
    avg_precision = total_precision / len(queries)
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        "recall": avg_recall,
        "precision": avg_precision,
        "f1": f1
    }

def run_lang_ablation(lang_code):
    data_dir = BASE_DATA_DIR / f"legal_{lang_code}"
    if lang_code == "it":
        data_dir = BASE_DATA_DIR / "italian_legal"
        
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"

    if not corpus_path.exists() or not queries_path.exists():
        print(f"Data for {lang_code} not found at {data_dir}.")
        return None, None

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    # 1. STEMMING
    searcher_stem = BM25Searcher(backend="python")
    tokenizer_stem = ForcedLanguageTokenizer(lang_code, use_stemming=True)
    searcher_stem._store.tokenizer = tokenizer_stem
    searcher_stem.tokenizer = tokenizer_stem
    searcher_stem.index_documents(docs)
    metrics_stem = calculate_metrics(searcher_stem, queries)

    # 2. NO STEMMING
    searcher_no_stem = BM25Searcher(backend="python")
    tokenizer_no_stem = ForcedLanguageTokenizer(lang_code, use_stemming=False)
    searcher_no_stem._store.tokenizer = tokenizer_no_stem
    searcher_no_stem.tokenizer = tokenizer_no_stem
    searcher_no_stem.index_documents(docs)
    metrics_no_stem = calculate_metrics(searcher_no_stem, queries)

    return metrics_no_stem, metrics_stem

def main():
    languages = ["it", "fr", "de", "es"]
    header = f"{'Lang':<5} | {'Metric':<10} | {'No Stem':<10} | {'With Stem':<10} | {'Delta':<10}"
    print(header)
    print("-" * len(header))
    
    for lang in languages:
        m_no, m_step = run_lang_ablation(lang)
        if m_no is not None:
            for metric in ["recall", "precision", "f1"]:
                v_no = m_no[metric]
                v_stem = m_step[metric]
                delta = v_stem - v_no
                print(f"{lang:<5} | {metric:<10} | {v_no:<10.4f} | {v_stem:<10.4f} | {delta:+.4f}")
            print("-" * len(header))

if __name__ == "__main__":
    main()
