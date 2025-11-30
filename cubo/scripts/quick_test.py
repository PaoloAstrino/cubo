from cubo.embeddings.model_loader import ModelManager
from cubo.retrieval.retriever import DocumentRetriever

model = ModelManager().load_model()
retriever = DocumentRetriever(model, use_sentence_window=True, use_auto_merging=False)

query = "tell me about the frog"
print(f"Query: {query}\n")

results = retriever.retrieve_top_documents(query, top_k=5)
for i, r in enumerate(results, 1):
    filename = r["metadata"]["filename"]
    sim = r["similarity"]
    base_sim = r.get("base_similarity", 0)
    preview = r["document"][:100]
    print(f"{i}. {filename} (sim: {sim:.4f}, base: {base_sim:.4f})")
    print(f"   {preview}...\n")
