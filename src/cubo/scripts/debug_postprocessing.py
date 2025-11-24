"""
Debug the postprocessing step to see what's happening to the chunks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cubo.embeddings.model_loader import ModelManager
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.security.security import security_manager

# Load model
model_manager = ModelManager()
model = model_manager.load_model()

# Initialize retriever
retriever = DocumentRetriever(model=model, use_sentence_window=True, use_auto_merging=False)

# Test query
query = "tell me about the frog"
print(f"Query: {security_manager.scrub(query)}\n")

# Get query embedding
query_embedding = retriever._generate_query_embedding(query)

# Query raw results from vector store (before postprocessing)
raw_results = retriever.collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=['documents', 'metadatas', 'distances']
)

print("=" * 80)
print("RAW VECTOR STORE RESULTS (before postprocessing)")
print("=" * 80)
for i, (doc, metadata, distance) in enumerate(zip(
    raw_results['documents'][0],
    raw_results['metadatas'][0],
    raw_results['distances'][0]
), 1):
    similarity = 1 - distance
    filename = metadata.get('filename', 'Unknown')
    sentence_idx = metadata.get('sentence_index', -1)
    window = metadata.get('window', '')[:200]

    print(f"\n{i}. {filename} (sentence {sentence_idx}, similarity: {similarity:.4f})")
    print(f"   Matched sentence: {doc[:150]}...")
    print(f"   Window preview: {window}...")

# Now get processed results
print("\n" + "=" * 80)
print("AFTER POSTPROCESSING")
print("=" * 80)

candidates = []
for doc, metadata, distance in zip(
    raw_results['documents'][0],
    raw_results['metadatas'][0],
    raw_results['distances'][0]
):
    candidates.append({
        "document": doc,
        "metadata": metadata,
        "similarity": 1 - distance
    })

# Apply postprocessing
processed = retriever._apply_window_postprocessing(candidates)

for i, result in enumerate(processed, 1):
    filename = result['metadata'].get('filename', 'Unknown')
    similarity = result.get('similarity', 0)
    doc_preview = result.get('document', '')[:200]

    print(f"\n{i}. {filename} (similarity: {similarity:.4f})")
    print(f"   Content: {doc_preview}...")
