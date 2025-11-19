"""
FastPass ingestion module
"""
from pathlib import Path
import os
import json
from typing import List
import pandas as pd

from src.cubo.ingestion.document_loader import DocumentLoader
from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.storage.metadata_manager import get_metadata_manager
from src.cubo.utils.logger import logger
from src.cubo.config import config


class FastPassIngestor:
    """Quick ingestion path to make documents queryable ASAP with BM25."""

    def __init__(self, output_dir: str = None, skip_model: bool = False):
        self.output_dir = Path(output_dir or config.get("ingestion.fast_pass.output_dir", config.get("fast_pass_output_dir", "data/fastpass")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DocumentLoader(skip_model=skip_model)
        self.skip_model = skip_model

    def ingest_folder(self, folder_path: str) -> dict:
        """Ingest a folder quickly and create a chunks parquet + BM25 stats.

        Returns: dict with paths {"chunks_parquet": path, "bm25_stats": path}
        """
        logger.info(f"Fast pass ingest start: {folder_path}")

        # Enforce skip model: clear enhanced processor if requested
        if self.skip_model:
            self.loader.enhanced_processor = None

        # Load chunks
        chunks = self.loader.load_documents_from_folder(folder_path)
        if not chunks:
            logger.warning("No chunks produced in fast pass ingest")
            return {}

        # Build DataFrame
        records = []
        texts = []
        doc_ids = []
        for c in chunks:
            filename = c.get('filename', 'unknown')
            file_hash = c.get('file_hash', '')
            chunk_index = c.get('chunk_index', 0)
            text = c.get('text', '') or c.get('document', '')
            token_count = c.get('token_count', len(text.split()))
            records.append({
                'filename': filename,
                'file_hash': file_hash,
                'chunk_index': chunk_index,
                'text': text,
                'token_count': token_count,
                'char_length': len(text)
            })
            texts.append(text)
            # create pseudo doc ids for BM25
            doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
            doc_ids.append(doc_id)

        df = pd.DataFrame.from_records(records)
        # Use newline-delimited JSON for portability (no pyarrow required)
        chunks_jsonl = self.output_dir / 'chunks.jsonl.tmp'
        final_chunks_jsonl = self.output_dir / 'chunks.jsonl'
        with open(chunks_jsonl, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # Atomic rename to avoid partial reads
        os.replace(str(chunks_jsonl), str(final_chunks_jsonl))
        logger.info(f"Saved {len(records)} chunks to {final_chunks_jsonl}")

        # Build BM25 stats by using BM25Searcher
        try:
            bm25 = BM25Searcher()
            docs = [{'doc_id': did, 'text': txt} for did, txt in zip(doc_ids, texts)]
            bm25.index_documents(docs)
            bm25_tmp = self.output_dir / 'bm25_stats.json.tmp'
            bm25_path = self.output_dir / 'bm25_stats.json'
            bm25.save_stats(str(bm25_tmp))
            os.replace(str(bm25_tmp), str(bm25_path))
            # record ingestion run in metadata db
            try:
                manager = get_metadata_manager()
                run_id = f"fastpass_{os.path.basename(str(Path(folder_path)))}_{int(pd.Timestamp.utcnow().timestamp())}"
                manager.record_ingestion_run(run_id, str(folder_path), len(records), str(final_chunks_jsonl))
            except Exception:
                logger.warning("Failed to record ingestion run to metadata DB")
        except Exception as e:
            logger.error(f"Failed to build BM25 stats: {e}")
            bm25_path = None

        # Write ingestion manifest for reproducibility
        manifest = {
            'ingested_at': pd.Timestamp.utcnow().isoformat(),
            'source_folder': folder_path,
            'chunks_count': len(records),
            'created_by': 'FastPassIngestor',
            'skip_model': self.skip_model
        }
        manifest_path = self.output_dir / 'ingestion_manifest.json.tmp'
        final_manifest_path = self.output_dir / 'ingestion_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        os.replace(str(manifest_path), str(final_manifest_path))

        return {
            'chunks_jsonl': str(final_chunks_jsonl),
            'bm25_stats': str(bm25_path) if bm25_path else None,
            'chunks_count': len(df)
        }


def build_bm25_index(folder_path: str, output_dir: str = None, skip_model: bool = False) -> dict:
    ingestor = FastPassIngestor(output_dir=output_dir, skip_model=skip_model)
    return ingestor.ingest_folder(folder_path)
