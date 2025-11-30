from pathlib import Path

from cubo.ingestion.ingestion_manager import IngestionManager
from cubo.storage.metadata_manager import get_metadata_manager


def test_start_fast_pass_creates_run(tmp_path: Path):
    sample_folder = Path(__file__).parent.parent.parent / "data"
    assert sample_folder.exists()
    manager = IngestionManager()
    res = manager.start_fast_pass(
        str(sample_folder), output_dir=str(tmp_path), skip_model=True, auto_deep=False
    )
    assert "run_id" in res
    run_id = res["run_id"]
    mm = get_metadata_manager()
    run = mm.get_ingestion_run(run_id)
    assert run is not None
    assert run["status"] in ["pending", "fast_complete"]
