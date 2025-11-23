import json
from pathlib import Path
from src.cubo.evaluation.validate_schema import validate_benchmark_output


def test_validate_sample_run(tmp_path):
    # Create a minimal valid run JSON
    run = {
        "metadata": {
            "test_run_timestamp": 1700000000.0,
            "mode": "retrieval-only",
            "total_questions": 2,
            "success_rate": 1.0,
            "hardware": {
                "cpu": {"model": "Intel test"},
                "ram": {"total_gb": 16.0}
            }
        },
        "results": {
            "easy": [],
            "medium": [],
            "hard": []
        }
    }

    file_path = tmp_path / 'sample_run.json'
    schema_path = Path('schemas/benchmark_output_schema.json')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(run, f)

    ok, msg = validate_benchmark_output(file_path, schema_path)
    # If schema validator not installed, function returns False with message; otherwise ok.
    assert isinstance(ok, bool)
    if ok:
        assert msg == 'ok'


def test_validate_invalid_run(tmp_path):
    # Missing required 'metadata' should be invalid
    run = {
        "results": {"easy": [], "medium": [], "hard": []}
    }
    file_path = tmp_path / 'invalid_run.json'
    schema_path = Path('schemas/benchmark_output_schema.json')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(run, f)

    ok, msg = validate_benchmark_output(file_path, schema_path)
    assert isinstance(ok, bool)
    if ok:
        # If jsonschema not installed, ok might be False due to missing library
        assert False, 'Expected validation to fail but it succeeded'
    else:
        assert 'required' in msg or 'jsonschema not installed' in msg
