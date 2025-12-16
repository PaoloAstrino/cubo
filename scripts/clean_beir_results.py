import os
import shutil
import glob
import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
ARCHIVE_ROOT = RESULTS_DIR / "archive"

# Patterns for files to clean up
PATTERNS = [
    "results/tonight_full/benchmark_beir*.json",
    "results/tonight_full/benchmark_ragas*.json",
    "results/tonight_full/analysis*.json",
    "results/tonight_full/top_bottom_examples.json",
    "results/tonight_full/*.log",
    "results/tonight_full/pipeline_stats.json",
    "results/beir_*_benchmark.json",
    "results/beir_*_test.json",
    "results/benchmark_results.json",
    "results/tonight_beir/*", 
]

def clean_beir_results():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE_ROOT / f"beir_run_{timestamp}"
    
    print(f"--- Cleaning BEIR results ---")
    print(f"Target archive: {archive_dir}")
    
    files_to_move = []
    for pattern in PATTERNS:
        # glob.glob returns strings, convert to Path
        matches = glob.glob(pattern)
        files_to_move.extend([Path(m) for m in matches])
    
    # Remove duplicates
    files_to_move = list(set(files_to_move))
    
    # Also check for storage dir in tonight_full
    storage_dir = RESULTS_DIR / "tonight_full" / "storage"
    
    if not files_to_move and not storage_dir.exists():
        print("No files or storage directory found to clean.")
        return

    if not archive_dir.exists():
        try:
            archive_dir.mkdir(parents=True)
        except Exception as e:
            print(f"Error creating archive directory: {e}")
            return

    # Move files
    for src in files_to_move:
        if src.is_file():
            dst = archive_dir / src.name
            # Handle duplicate names if flattening structure (simple approach: just move)
            # If multiple patterns match files in different dirs with same name, this might overwrite.
            # Given the patterns, most are in tonight_full or root results.
            # Let's preserve parent dir name if it's not tonight_full to avoid collisions?
            # For now, simple move.
            print(f"Archiving: {src}")
            try:
                shutil.move(str(src), str(dst))
            except Exception as e:
                print(f"Error moving {src}: {e}")

    # Move storage directory
    if storage_dir.exists():
        dst_storage = archive_dir / "storage"
        print(f"Archiving storage: {storage_dir}")
        try:
            shutil.move(str(storage_dir), str(dst_storage))
        except Exception as e:
            print(f"Error moving storage: {e}")
            
    print(f"--- Cleanup complete. Files moved to {archive_dir} ---")

if __name__ == "__main__":
    clean_beir_results()
