"""Utility script to clear the FAISS vector store."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cubo.config import config
from src.cubo.retrieval.vector_store import create_vector_store


def parse_args():
    parser = argparse.ArgumentParser(description="Clear FAISS vector store contents")
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--index-dir', default=None, help='Override FAISS directory')
    parser.add_argument('--wipe-storage', action='store_true', help='Remove on-disk storage after reset')
    return parser.parse_args()


def _delete_path(path: Path):
    if path.exists():
        print(f"\nDeleting {path}...")
        shutil.rmtree(path)
        print("✓ Storage folder removed")
    else:
        print(f"\n{path} doesn't exist, nothing to delete")


def main():
    args = parse_args()
    print("=" * 80)
    print("CLEARING FAISS VECTOR STORE")
    print("=" * 80)

    if args.collection:
        config.set('collection_name', args.collection)
    if args.index_dir:
        config.set('vector_store_path', args.index_dir)

    store = create_vector_store(
        collection_name=args.collection,
        index_dir=args.index_dir
    )

    reset_fn = getattr(store, 'reset', None)
    if callable(reset_fn):
        reset_fn()
        print("\nFAISS index reset successfully.")
    else:
        print("\nVector store reset not supported; manual cleanup may be required.")

    if args.wipe_storage:
        path = Path(config.get('vector_store_path', './faiss_index'))
        _delete_path(path)

    print("\n" + "=" * 80)
    print("DATABASE CLEARED - Please re-upload your documents in the GUI")
    print("=" * 80)
    print("\nAfter re-uploading, retrieval should work correctly!")


if __name__ == '__main__':
    main()

