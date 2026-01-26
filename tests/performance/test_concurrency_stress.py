import concurrent.futures

from cubo.retrieval.vector_store import FaissStore


def test_concurrent_writes_sqlite_locking(tmp_path):
    """
    Stress Test: Verify if SQLite back-end handles concurrent writes or locks up.

    Scenario:
    - 10 threads try to add documents simultaneously.
    - SQLite (default) often fails with 'database is locked' if not handled correctly via WAL or timeout.
    """
    store_dir = tmp_path / "stress_store"
    store = FaissStore(dimension=384, index_dir=store_dir)

    # Track results
    success_count = 0
    errors = []

    def add_doc(i):
        nonlocal success_count
        try:
            # Create a dummy vector and doc
            vector = [0.1] * 384
            doc_id = f"doc_{i}"
            store.add(
                embeddings=[vector], documents=[f"Content {i}"], metadatas=[{"id": i}], ids=[doc_id]
            )
            return True
        except Exception as e:
            errors.append(str(e))
            return False

    # Run concurrent adds
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(add_doc, i) for i in range(50)]
        results = [f.result() for f in futures]

    successes = results.count(True)

    # We EXPECT this to potentially fail or show locking issues in a naive implementation
    print(f"\nSuccess: {successes}/50")
    print(f"Errors: {len(errors)}")
    if errors:
        print(f"Sample error: {errors[0]}")

    # Assert basic consistency - if we lost data, that's a problem
    # If we crashed with "database locked", that confirms the stress test "passed" (it revealed the flaw)
    # But for a test suite, we generally want to assert the *expected behavior*.
    # Here, we assert that the system is ROBUST (so it *should* pass ideally).
    # If it fails, pytest will report failure, which is what we want to verify the flaw.

    assert successes == 50, f"Failed to handle concurrency. Errors: {errors[:3]}"
    assert store.count() == 50, f"Store count mismatch. Expected 50, got {store.count()}"
