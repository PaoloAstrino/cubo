#!/usr/bin/env python3
"""
Download and setup Dolphin model for CUBO integration
Downloads ByteDance/Dolphin to models/dolphin/ folder
"""

import logging
from pathlib import Path

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dolphin_model():
    """Download Dolphin model to models/dolphin/ folder."""
    dolphin_dir = _setup_directories()

    try:
        tokenizer, processor, model = _download_model_components(dolphin_dir)
        _save_model_locally(dolphin_dir, tokenizer, processor, model)
        _verify_and_report_download(dolphin_dir)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def _setup_directories():
    """Setup and create the model directories."""
    models_dir = Path("./models")
    dolphin_dir = models_dir / "dolphin"

    print("🐬 Downloading ByteDance/Dolphin model...")
    print(f"Target directory: {dolphin_dir.absolute()}")

    # Create directory if it doesn't exist
    dolphin_dir.mkdir(parents=True, exist_ok=True)
    return dolphin_dir


def _resolve_hf_revision():
    """Resolve a pinned HF revision from env var or enforce explicit opt-in.

    Behavior:
      - If HF_PINNED_REVISION is set, return it and use it for `revision`.
      - If HF_ALLOW_UNPINNED_HF_DOWNLOADS=1 is set, allow unpinned downloads (warning).
      - Otherwise raise a RuntimeError to force explicit pinning or opt-in.
    """
    import os

    rev = os.getenv("HF_PINNED_REVISION")
    allow_unpinned = os.getenv("HF_ALLOW_UNPINNED_HF_DOWNLOADS", "0") == "1"
    if rev:
        return rev
    if allow_unpinned:
        logger.warning(
            "Hugging Face downloads are unpinned (HF_ALLOW_UNPINNED_HF_DOWNLOADS=1)."
            " For security, prefer pinning via HF_PINNED_REVISION."
        )
        return None
    raise RuntimeError(
        "Hugging Face downloads must be pinned. Set HF_PINNED_REVISION or set HF_ALLOW_UNPINNED_HF_DOWNLOADS=1 to opt-in."
    )


def _download_model_components(dolphin_dir):
    """Download tokenizer, processor, and model components."""
    print("📥 Downloading tokenizer...")
    revision = _resolve_hf_revision()
    kwargs = {"cache_dir": str(dolphin_dir)}
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained("ByteDance/Dolphin", **kwargs)  # nosec

    print("📥 Downloading processor...")
    processor = AutoProcessor.from_pretrained("ByteDance/Dolphin", **kwargs)  # nosec

    print("📥 Downloading model (400MB)...")
    model_kwargs = {"cache_dir": str(dolphin_dir), "torch_dtype": "auto"}
    if revision:
        model_kwargs["revision"] = revision
    model = AutoModelForVision2Seq.from_pretrained(
        "ByteDance/Dolphin", **model_kwargs
    )  # nosec

    return tokenizer, processor, model


def _save_model_locally(dolphin_dir, tokenizer, processor, model):
    """Save all model components to local directory."""
    print("💾 Saving model locally...")
    tokenizer.save_pretrained(dolphin_dir)
    processor.save_pretrained(dolphin_dir)
    model.save_pretrained(dolphin_dir)

    print("✅ Dolphin model downloaded successfully!")
    print(f"📁 Model saved to: {dolphin_dir}")


def _verify_and_report_download(dolphin_dir):
    """Verify download and report file information."""
    files = list(dolphin_dir.glob("*"))
    print(f"📊 Downloaded {len(files)} files:")
    for file in sorted(files):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.1f} MB")


def test_dolphin_model():
    """Test that the downloaded model works."""

    models_dir = Path("./models")
    dolphin_dir = models_dir / "dolphin"

    if not dolphin_dir.exists():
        print("❌ Dolphin model not found. Run download first.")
        return False

    try:
        print("🧪 Testing Dolphin model...")

        # Load from local directory (safe)
        # Explicitly mark as nosec for Bandit (local load)
        tokenizer = AutoTokenizer.from_pretrained(dolphin_dir)  # nosec
        processor = AutoProcessor.from_pretrained(dolphin_dir)  # nosec
        model = AutoModelForVision2Seq.from_pretrained(dolphin_dir)  # nosec

        # Verify components loaded
        assert tokenizer is not None
        assert processor is not None
        assert model is not None

        print("✅ Model loaded successfully from local directory!")
        print("🎯 Ready for integration with EmbeddingGemma-300M")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Dolphin model for CUBO")
    parser.add_argument("--download", action="store_true", help="Download the model")
    parser.add_argument("--test", action="store_true", help="Test the downloaded model")

    args = parser.parse_args()

    if args.download:
        success = download_dolphin_model()
        if success and not args.test:
            print("\n💡 Tip: Run with --test to verify the download")
    elif args.test:
        success = test_dolphin_model()
    else:
        print("Usage:")
        print("  python download_dolphin.py --download  # Download model")
        print("  python download_dolphin.py --test      # Test downloaded model")
        print("  python download_dolphin.py --download --test  # Download and test")
