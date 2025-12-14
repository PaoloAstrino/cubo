"""
Model Inference Threading for CUBO
Provides threaded model inference with proper GPU/CPU utilization.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

import torch

from cubo.utils.logger import logger


class ModelInferenceThreading:
    """Thread-safe model inference with proper GPU utilization and batching."""

    def __init__(self, max_concurrent_models: int = 2, gpu_memory_limit: float = 0.8):
        """
        Initialize model inference threading.

        Args:
            max_concurrent_models: Maximum concurrent model operations
            gpu_memory_limit: GPU memory usage limit (0.0-1.0)
        """
        self.max_concurrent = max_concurrent_models
        self.gpu_memory_limit = gpu_memory_limit

        # Thread synchronization
        self.model_semaphore = threading.Semaphore(max_concurrent_models)
        self.gpu_lock = threading.Lock()

        # GPU memory management
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"GPU detected: {torch.cuda.get_device_name(0)} "
                f"({self.total_gpu_memory / 1024**3:.1f}GB)"
            )

        # Performance tracking
        self.inference_stats = {
            "embeddings_generated": 0,
            "dolphin_inferences": 0,
            "total_embedding_time": 0.0,
            "total_dolphin_time": 0.0,
        }

    def generate_embeddings_threaded(
        self, texts: List[str], embedding_model, batch_size: int = 8, timeout_per_batch: int = 600
    ) -> List[List[float]]:
        """
        Generate embeddings using threaded batching.

        Args:
            texts: List of texts to embed
            embedding_model: Embedding model instance
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()
        logger.info(f"Starting threaded embedding generation for {len(texts)} texts")

        # Split into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(f"Split into {len(batches)} batches of max size {batch_size}")

        # Process batches in parallel
        all_embeddings = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(self._embed_batch, batch, embedding_model): batch
                for batch in batches
            }

            # Collect results in order

            for future in as_completed(future_to_batch, timeout=timeout_per_batch):
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Add empty embeddings for failed batch
                    batch = future_to_batch[future]
                    all_embeddings.extend([[] for _ in batch])
            # If some futures still pending, cancel them and handle
            pending = [f for f in future_to_batch.keys() if not f.done()]
            if pending:
                for p in pending:
                    try:
                        p.cancel()
                    except Exception:
                        pass
                logger.warning(
                    f"{len(pending)} embedding tasks did not complete within timeout ({timeout_per_batch}s) and were cancelled"
                )

        # Update stats
        total_time = time.time() - start_time
        self.inference_stats["embeddings_generated"] += len(texts)
        self.inference_stats["total_embedding_time"] += total_time

        # Avoid division by zero when execution was extremely fast
        if total_time > 0:
            rate = f"{len(texts)/total_time:.1f}"
        else:
            rate = "N/A"
        logger.info(
            f"Threaded embedding completed: {len(all_embeddings)} embeddings "
            f"in {total_time:.2f}s ({rate} texts/sec)"
        )

        return all_embeddings

    def _embed_batch(self, text_batch: List[str], embedding_model) -> List[List[float]]:
        """Embed a single batch with semaphore control."""
        with self.model_semaphore:  # Limit concurrent model usage
            try:
                # GPU memory check
                if self.gpu_available and self._check_gpu_memory():
                    torch.cuda.empty_cache()  # Free memory if needed

                # Generate embeddings
                try:
                    embeddings = embedding_model.encode(text_batch, batch_size=len(text_batch))
                except TypeError:
                    # Some mock models or older APIs may not accept batch_size kwarg
                    embeddings = embedding_model.encode(text_batch)

                # Convert to list if needed
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()

                return embeddings

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                return [[] for _ in text_batch]

    def run_dolphin_inference_threaded(
        self, images_or_texts: List[Any], dolphin_processor, batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Run Dolphin inference with threading.

        Args:
            images_or_texts: List of images or texts for processing
            dolphin_processor: Dolphin processor instance
            batch_size: Batch size (usually 1 for GPU memory)

        Returns:
            List of processing results
        """
        if not images_or_texts:
            return []

        start_time = time.time()
        logger.info(f"Starting threaded Dolphin inference for {len(images_or_texts)} items")

        batches = [
            images_or_texts[i : i + batch_size] for i in range(0, len(images_or_texts), batch_size)
        ]

        all_results = []
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, 2)) as executor:
            # Submit GPU tasks
            future_to_batch = {
                executor.submit(self._process_dolphin_batch, batch, dolphin_processor): batch
                for batch in batches
            }

            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Dolphin batch processing failed: {e}")
                    batch = future_to_batch[future]
                    all_results.extend([{"error": str(e)} for _ in batch])

        # Update stats
        total_time = time.time() - start_time
        self.inference_stats["dolphin_inferences"] += len(images_or_texts)
        self.inference_stats["total_dolphin_time"] += total_time

        logger.info(
            f"Threaded Dolphin inference completed: {len(all_results)} results "
            f"in {total_time:.2f}s"
        )

        return all_results

    def _process_dolphin_batch(self, batch: List[Any], dolphin_processor) -> List[Dict[str, Any]]:
        """Process a batch with Dolphin."""
        results = []

        for item in batch:
            with self.model_semaphore:
                with self.gpu_lock:  # Exclusive GPU access for Dolphin
                    try:
                        # GPU memory management
                        if self.gpu_available:
                            self._manage_gpu_memory()

                        # Process item
                        result = dolphin_processor.process_item(item)
                        results.append(result)

                    except Exception as e:
                        logger.error(f"Dolphin processing failed: {e}")
                        results.append({"error": str(e)})

        return results

    def _check_gpu_memory(self) -> bool:
        """Check if GPU memory usage is too high."""
        if not self.gpu_available:
            return False

        try:
            allocated = torch.cuda.memory_allocated()
            memory_usage = allocated / self.total_gpu_memory
            return memory_usage > self.gpu_memory_limit
        except Exception:
            return False

    def _manage_gpu_memory(self):
        """Manage GPU memory usage."""
        if not self.gpu_available:
            return

        try:
            # Check memory usage
            if self._check_gpu_memory():
                logger.warning("GPU memory limit exceeded, clearing cache")
                torch.cuda.empty_cache()

                # Wait a bit for memory to be freed
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"GPU memory management failed: {e}")

    def run_inference_threaded(
        self, inference_fn: Callable, inputs: List[Any], batch_size: int = 4, use_gpu: bool = False
    ) -> List[Any]:
        """
        Generic threaded inference function.

        Args:
            inference_fn: Function to run inference
            inputs: List of inputs
            batch_size: Batch size
            use_gpu: Whether to use GPU lock

        Returns:
            List of results
        """
        if not inputs:
            return []

        # Split into batches
        batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]

        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_batch = {
                executor.submit(self._run_batch_inference, inference_fn, batch, use_gpu): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    batch = future_to_batch[future]
                    all_results.extend([None for _ in batch])

        return all_results

    def _run_batch_inference(
        self, inference_fn: Callable, batch: List[Any], use_gpu: bool
    ) -> List[Any]:
        """Run inference on a batch."""
        results = []

        for item in batch:
            if use_gpu:
                with self.gpu_lock:
                    self._manage_gpu_memory()
                    result = inference_fn(item)
            else:
                with self.model_semaphore:
                    result = inference_fn(item)

            results.append(result)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        stats = self.inference_stats.copy()

        # Calculate rates
        if stats["total_embedding_time"] > 0:
            stats["embedding_rate"] = stats["embeddings_generated"] / stats["total_embedding_time"]

        if stats["total_dolphin_time"] > 0:
            stats["dolphin_rate"] = stats["dolphin_inferences"] / stats["total_dolphin_time"]

        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_stats = {
            "embeddings_generated": 0,
            "dolphin_inferences": 0,
            "total_embedding_time": 0.0,
            "total_dolphin_time": 0.0,
        }


# Global instance
_model_inference_threading = None


def get_model_inference_threading() -> ModelInferenceThreading:
    """Get the global model inference threading instance."""
    global _model_inference_threading
    if _model_inference_threading is None:
        _model_inference_threading = ModelInferenceThreading()
    return _model_inference_threading
