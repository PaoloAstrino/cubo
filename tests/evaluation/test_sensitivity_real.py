import importlib.util
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "sensitivity_analysis_real",
    Path(__file__).parents[1] / ".." / "tools" / "sensitivity_analysis_real.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
measure_faiss_sensitivity_real = mod.measure_faiss_sensitivity_real


class FakeRetriever:
    def __init__(self, ret):
        self._ret = ret
        self.collection = types.SimpleNamespace(index=None)

    def retrieve(self, q, top_k=10):
        return self._ret


def test_measure_handles_non_list_results():
    fake = FakeRetriever(ret=42)  # non-iterable/sentinel value
    queries = [{"text": "what is cubo"}]
    res = measure_faiss_sensitivity_real(fake, queries, nprobe=1, top_k=5)
    # should return empty dict because no valid latencies were recorded
    assert res == {}
