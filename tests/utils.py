import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.indexing.index_publisher import publish_version


def make_vectors(n: int, dim: int = 2, base: float = 0.0):
    return [[float(i + j + base) for j in range(dim)] for i in range(n)]


def create_and_publish_faiss_index(index_root: Path, vname: str, n_vectors: int = 16, dim: int = 2):
    vdir = Path(index_root) / vname
    manager = FAISSIndexManager(dimension=dim, index_dir=vdir)
    vectors = make_vectors(n_vectors, dim)
    ids = [f"id_{i}" for i in range(len(vectors))]
    manager.build_indexes(vectors, ids)
    manager.save(path=vdir)
    publish_version(vdir, index_root)
    return vdir, vectors, ids


def reader_loop(
    manager: FAISSIndexManager, query: List[float], iterations: int = 100, delay: float = 0.01
):
    errors = []
    results = []
    for _ in range(iterations):
        try:
            r = manager.search(query, k=1)
            results.append(r)
        except Exception as exc:
            errors.append(exc)
        time.sleep(delay)
    return results, errors


def run_script(script: Path, args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    env = dict()
    # Use the existing environment to pass PYTHONPATH and other vars
    env.update({})
    process = subprocess.run(
        [str(script)] + args, cwd=cwd or Path.cwd(), capture_output=True, text=True, env=None
    )
    return process.returncode, process.stdout, process.stderr
