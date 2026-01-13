import sys
import os
from pathlib import Path

# Set project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

print("=== CUBO BM25 Tokenizer Diagnostic ===")

# 1. Check nltk
try:
    import nltk
    print(f"[OK] nltk version {nltk.__version__} installed.")
except ImportError:
    print("[FAIL] nltk is NOT installed.")

# 2. Check simplemma
try:
    import simplemma
    print(f"[OK] simplemma installed.")
except ImportError:
    print("[INFO] simplemma not installed (optional for lemmatization).")

# 3. Test MultilingualTokenizer
try:
    from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer
    tokenizer = MultilingualTokenizer(use_stemming=True)
    print("[OK] MultilingualTokenizer initialized successfully.")
    
    # Test English Stemming
    en_text = "The quick brown foxes are jumping"
    en_tokens = tokenizer.tokenize(en_text, language="en")
    print(f"English: '{en_text}' -> {en_tokens}")
    
    # Test Italian Stemming (The 'Europe' test)
    it_text = "I gatti mangiano velocemente"
    it_tokens = tokenizer.tokenize(it_text, language="it")
    print(f"Italian: '{it_text}' -> {it_tokens}")
    
    if "fox" in en_tokens or "foxes" not in en_tokens:
        print("[OK] Stemming appears active.")
    else:
        print("[WARN] Tokens returned but stemming might be inactive.")

except Exception as e:
    print(f"[CRITICAL] MultilingualTokenizer failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Check BM25 Configuration logic
from cubo.config import config
print(f"\n=== Configuration Check ===")
print(f"Laptop Mode: {config.get('laptop_mode')}")
print(f"BM25 Use Multilingual (Implicit): {config.get('bm25.use_multilingual', 'Not Set')}")
print(f"BM25 Use Lemmatization: {config.get('bm25.use_lemmatization', 'Not Set')}")
