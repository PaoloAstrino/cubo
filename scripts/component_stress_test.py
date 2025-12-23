"""
Component Stress Test & Speed Profiler for CUBO

This script runs isolated performance tests on key system components to
benchmark local hardware capabilities. It measures throughput and latency
for:
1. Text Chunking
2. Embedding Generation (CPU/GPU)
3. Vector Search (FAISS)
4. BM25 Indexing
5. LLM Generation (Tokens/sec)

Usage:
    python scripts/component_stress_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.retrieval.bm25_python_store import BM25PythonStore
from cubo.utils.utils import Utils

# Try to import FAISS
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def generate_random_text(num_sentences=100) -> str:
    """Generate random text for testing."""
    words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "neural",
        "network",
        "deep",
        "python",
        "code",
        "programming",
        "algorithm",
        "data",
        "structure",
        "system",
    ]

    text = []
    for _ in range(num_sentences):
        sentence = " ".join(np.random.choice(words, 10)) + "."
        text.append(sentence)
    return " ".join(text)


def test_chunking_speed(size_mb=1):
    """Test text chunking speed."""
    print(f"\n--- Testing Chunking Speed ({size_mb}MB text) ---")

    # Generate ~1MB of text
    # 1 char ~= 1 byte. 1MB = 10^6 bytes.
    # A sentence is ~60 bytes. Need ~16,000 sentences.
    n_sentences = int((size_mb * 1_000_000) / 60)
    print(f"Generating {size_mb}MB of random text...")
    text = generate_random_text(n_sentences)
    total_chars = len(text)
    print(f"Text size: {total_chars / 1_000_000:.2f} MB")

    start = time.time()
    chunks = Utils.chunk_text(text, chunk_size=1000, chunk_overlap=200)
    duration = time.time() - start

    print(f"Chunked into {len(chunks)} chunks in {duration:.4f}s")
    print(f"Speed: {total_chars / duration / 1_000_000:.2f} MB/s")
    return duration


def test_embedding_speed(batch_size=32, n_sentences=100):
    """Test embedding generation speed."""
    print(f"\n--- Testing Embedding Speed (Batch: {batch_size}, N: {n_sentences}) ---")

    try:
        embedder = EmbeddingGenerator()
        print(f"Model: {config.get('model_path')}")
    except Exception as e:
        print(f"Skipping embedding test: {e}")
        return

    sentences = [generate_random_text(1) for _ in range(n_sentences)]

    # Warmup
    embedder.encode(sentences[:5])

    start = time.time()
    embeddings = embedder.encode(sentences, batch_size=batch_size)
    duration = time.time() - start

    # embeddings is a list of lists or numpy array
    n_embeds = len(embeddings)

    print(f"Generated {n_embeds} embeddings in {duration:.4f}s")
    print(f"Throughput: {n_embeds / duration:.2f} sentences/sec")
    print(f"Latency per sentence: {duration / n_embeds * 1000:.2f} ms")
    return duration


def test_faiss_speed(n_vectors=10000, dim=384):
    """Test FAISS search speed."""
    print(f"\n--- Testing FAISS Search Speed (N: {n_vectors}, Dim: {dim}) ---")

    if not HAS_FAISS:
        print("FAISS not installed. Skipping.")
        return

    # Generate random vectors
    data = np.random.random((n_vectors, dim)).astype("float32")

    # Build Index
    index = faiss.IndexFlatL2(dim)
    start_build = time.time()
    index.add(data)
    build_time = time.time() - start_build
    print(f"Index build time: {build_time:.4f}s")

    # Search
    query = np.random.random((1, dim)).astype("float32")
    start_search = time.time()
    distances, indices = index.search(query, k=10)
    search_time = time.time() - start_search

    print(f"Search time (1 query): {search_time * 1000:.4f} ms")

    # Batch Search
    queries = np.random.random((100, dim)).astype("float32")
    start_batch = time.time()
    distances, indices = index.search(queries, k=10)
    batch_time = time.time() - start_batch

    print(f"Batch search time (100 queries): {batch_time:.4f}s")
    print(f"QPS: {100 / batch_time:.2f}")


def test_bm25_speed(n_docs=5000):
    """Test BM25 indexing and search speed."""
    print(f"\n--- Testing BM25 Speed (N: {n_docs} docs) ---")

    docs = []
    for i in range(n_docs):
        docs.append({"doc_id": str(i), "text": generate_random_text(5)})

    store = BM25PythonStore()

    start_index = time.time()
    store.add_documents(docs)
    index_time = time.time() - start_index
    print(f"Indexing time: {index_time:.4f}s")
    print(f"Indexing rate: {n_docs / index_time:.2f} docs/sec")

    start_search = time.time()
    store.search("apple banana", top_k=10)
    search_time = time.time() - start_search
    print(f"Search time: {search_time * 1000:.4f} ms")


def test_llm_generation_speed():
    """Test Local LLM generation speed."""
    print("\n--- Testing LLM Generation Speed ---")

    from cubo.processing.llm_local import LocalResponseGenerator

    try:
        llm = LocalResponseGenerator()
        print(f"Model loaded: {llm.model_path}")
    except Exception as e:
        print(f"Skipping LLM test: {e}")
        return

    query = "Write a short poem about coding."
    context = "Programming is the art of telling a computer what to do."

    start = time.time()
    response = llm.generate_response(query, context)
    duration = time.time() - start

    # Estimate tokens (rough approximation)
    tokens = len(response.split()) * 1.3

    print(f"Response generated in {duration:.4f}s")
    print(f"Response length: ~{int(tokens)} tokens")
    print(f"Speed: {tokens / duration:.2f} tokens/sec")


if __name__ == "__main__":
    print("Starting Component Stress Test...")
    print("=" * 40)

    test_chunking_speed(size_mb=1)
    test_embedding_speed(batch_size=32, n_sentences=100)
    test_faiss_speed(n_vectors=10000)
    test_bm25_speed(n_docs=5000)
    test_llm_generation_speed()

    print("\n" + "=" * 40)
    print("Test Complete.")
