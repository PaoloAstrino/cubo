.PHONY: ablation reranker system update-report

ablation:
	python scripts/run_ablation.py --dataset nfcorpus --top-k 50

reranker:
	python scripts/run_reranker_eval.py --index-dir results/beir_index_nfcorpus_hybrid_topk50 --queries data/beir/nfcorpus/queries.jsonl --top-k 50

system:
	python scripts/system_metrics.py --corpus data/beir/nfcorpus/corpus.jsonl --index-dir results/beir_index_nfcorpus_sys --queries data/beir/nfcorpus/queries.jsonl --top-k 50 --limit 200

update-report:
	python scripts/update_evaluation_report.py --ablation results --reranker results/reranker_eval_topk50.json --system results/system_metrics_*.json
