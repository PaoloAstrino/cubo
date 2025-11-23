import json
import os
import tempfile
import shutil
from pathlib import Path
import pytest

from scripts.benchmark_runner import BenchmarkRunner


def create_sample_dataset(tmpdir: str):
    data_folder = Path(tmpdir) / "sample_data"
    data_folder.mkdir(parents=True, exist_ok=True)
    # Create a few sample text files
    (data_folder / "doc1.txt").write_text("This is a test document about agriculture and crops.")
    (data_folder / "doc2.txt").write_text("This is a test document about computer science and algorithms.")

    # Questions JSON
    questions = {
        "metadata": {"total_questions": 2},
        "questions": {
            "easy": ["What is crop rotation?", "What is a binary search algorithm?"],
            "medium": [],
            "hard": []
        }
    }
    qpath = Path(tmpdir) / "questions.json"
    qpath.write_text(json.dumps(questions))

    # Ground truth
    ground_truth = {
        "easy_1": ["doc1"],
        "easy_2": ["doc2"]
    }
    gtpath = Path(tmpdir) / "ground_truth.json"
    gtpath.write_text(json.dumps(ground_truth))

    return str(data_folder), str(qpath), str(gtpath)


def test_benchmark_runner_retrieval_only(tmp_path):
    # Skip test on environments missing heavy ML dependencies
    pytest.importorskip('sentence_transformers')
    # Setup temp structure
    base_dir = str(tmp_path)
    data_folder, questions_path, ground_truth = create_sample_dataset(base_dir)

    # Retrieval configs (in-memory)
    retrieval_configs = [
        {"name": "hybrid_test", "config_updates": {"vector_store_backend": "faiss"}}
    ]
    ablations = [{"name": "none", "config_updates": {}}]

    # Initialize runner and run
    runner = BenchmarkRunner(
        datasets=[{"path": data_folder, "name": "sample", "questions": questions_path, "ground_truth": ground_truth, "easy_limit": 1}],
        retrieval_configs=retrieval_configs,
        ablations=ablations,
        k_values=[5, 10],
        mode="retrieval-only",
        output_dir=os.path.join(base_dir, "results")
    )

    # Avoid running ingestion in the test environment to prevent heavy dependency failures
    runner.run(run_ingest_first=False)

    # Assert results saved
    results_dir = Path(base_dir) / "results"
    assert results_dir.exists()

    # There should be a summary CSV
    summary_csv = results_dir / "summary.csv"
    assert summary_csv.exists()
    # At least one run folder exists
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) > 0

    # Check that each run has a benchmark_run.json file
    found = False
    for rd in run_dirs:
        if (rd / "benchmark_run.json").exists():
            found = True
            break
    assert found

    # Cleanup created files
    shutil.rmtree(results_dir)
