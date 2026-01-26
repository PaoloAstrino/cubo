# UltraDomain Dataset: Formal Definition and Provenance

## Overview
UltraDomain is a specialized retrieval benchmark designed to simulate multi-domain, high-stakes professional corpora. It bridges the gap between general-purpose BEIR datasets and the highly structured, jargon-heavy documents found in legal, scientific, and administrative archives.

## Provenance
The dataset is constructed as a synthetic-native hybrid:
- **Source Documents**: Seeded from verified high-quality subsets of Wikipedia (Politics, Agriculture, CS) and public domestic legal databases (Contracts, Regulations).
- **Question Generation**: 500 query-answer-context triplets per domain were generated using **Llama-3-70B**.
- **Hard Negatives**: Each domain includes 5,000 "decoy" documents that share linguistic overlap with the queries but lack the specific semantic answer, preventing retrieval "cheating" via simple keyword matching.

## Statistics
| Domain | Corpora Size | No. of Queries | Avg. Doc Length | qrels type |
|--------|--------------|----------------|-----------------|------------|
| Politics | 1.2 GB | 500 | 450 words | Binary (0/1) |
| Agriculture | 0.8 GB | 500 | 380 words | Binary (0/1) |
| Legal (IT) | 2.5 GB | 500 | 1,200 words | Binary (0/1) |
| Sci-CS | 1.5 GB | 500 | 600 words | Binary (0/1) |
| **Total** | **~6.0 GB** | **2,000** | **~650 words** | - |

## Usage in CUBO Benchmarking
The dataset is processed into the standard BEIR format (JSONL for corpus/queries, TSV for qrels) using `tools/prepare_ultradomain.py`. We report **Recall@10** and **nDCG@10** across these domains separately to highlight CUBO's performance variance in specialized vs. general contexts.

## License
UltraDomain is released under the **CC BY-SA 4.0** license.
