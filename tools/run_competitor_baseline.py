import os
import time
import json
import psutil
from pathlib import Path
from typing import List, Dict

import numpy as np
from llama_index.core import VectorStoreIndex, Document

# Use simple local embeddings to avoid dependency hell
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def run_llamaindex_baseline(corpus_path: str, queries_path: str, qrels_path: str, top_k=10):
    print(f"--- Running LlamaIndex Baseline ---")
    
    # 1. Load Data
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000: break # Small subset for rapid validation
            data = json.loads(line)
            documents.append(Document(text=data.get("text", ""), doc_id=data.get("_id")))
    
    # 2. Setup Index (Default SimpleVectorStore)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    start_time = time.time()
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    print(f"Indexing {len(documents)} documents using default SimpleVectorStore...")
    index = VectorStoreIndex.from_documents(
        documents, 
        embed_model=embed_model,
        show_progress=True
    )
    
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    print(f"Ingestion Time: {end_time - start_time:.2f}s")
    print(f"Peak RAM during ingest: {end_mem:.2f} MB (Delta: {end_mem - start_mem:.2f} MB)")
    
    # 3. Retrieve
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50: break
            queries.append(json.loads(line))
            
    print(f"Retrieving for {len(queries)} queries...")
    latencies = []
    hits = 0
    
    # Load qrels for evaluation
    qrels = {}
    with open(qrels_path, "r") as f:
        next(f)
        for line in f:
            qid, did, score = line.strip().split("\t")
            if qid not in qrels: qrels[qid] = []
            if int(score) > 0: qrels[qid].append(did)

    for q in queries:
        q_start = time.time()
        response = query_engine.query(q["text"])
        latencies.append(time.time() - q_start)
        
        retrieved_ids = [node.node.node_id for node in response.source_nodes]
        relevant_ids = qrels.get(q["_id"], [])
        if any(rid in relevant_ids for rid in retrieved_ids):
            hits += 1
            
    avg_latency = np.mean(latencies) * 1000
    recall = hits / len(queries)
    
    print(f"Avg Retrieval Latency: {avg_latency:.2f} ms")
    print(f"Recall@{top_k}: {recall:.4f}")
    
    return {
        "ingest_time": end_time - start_time,
        "peak_ram_mb": end_mem,
        "avg_latency_ms": avg_latency,
        "recall": recall
    }

if __name__ == "__main__":
    # Example for FiQA (if available)
    corpus = "data/beir/fiqa/corpus.jsonl"
    queries = "data/beir/fiqa/queries.jsonl"
    qrels = "data/beir/fiqa/qrels/test.tsv"
    
    if Path(corpus).exists():
        run_llamaindex_baseline(corpus, queries, qrels)
    else:
        print("Data not found. Please run with paths to a BEIR dataset.")
