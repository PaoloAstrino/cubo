import json
import os

from scripts.calculate_beir_metrics import calculate_metrics


def test_calculate_metrics_ndcg(tmp_path):
    # Create a fake results JSON
    run_path = tmp_path / "run.json"
    data = {"1": {"d1": 0.9, "d2": 0.5}, "2": {"d3": 0.8, "d4": 0.2}}
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Create a fake qrels TSV
    qrels_path = tmp_path / "qrels.tsv"
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        f.write("1\td1\t1\n")
        f.write("1\td2\t0\n")
        f.write("2\td3\t1\n")
        f.write("2\td4\t0\n")

    # Run metrics calculation
    calculate_metrics(str(run_path), str(qrels_path), k=2)

    # Check that metrics file was created
    metrics_file = str(run_path).replace(".json", "_metrics_k2.json")
    assert os.path.exists(metrics_file)
    with open(metrics_file, "r", encoding="utf-8") as mf:
        metrics = json.load(mf)
    assert "ndcg" in metrics
    assert metrics["k"] == 2
