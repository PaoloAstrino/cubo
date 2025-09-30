#!/usr/bin/env python3
"""
Download and setup Dolphin model for CUBO integration
Downloads ByteDance/Dolphin to models/dolphin/ folder
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dolphin_model():
    """Download Dolphin model to models/dolphin/ folder."""

    # Setup paths
    models_dir = Path("./models")
    dolphin_dir = models_dir / "dolphin"

    print("🐬 Downloading ByteDance/Dolphin model...")
    print(f"Target directory: {dolphin_dir.absolute()}")

    try:
        # Create directory if it doesn't exist
        dolphin_dir.mkdir(parents=True, exist_ok=True)

        # Download model components
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "ByteDance/Dolphin",
            cache_dir=str(dolphin_dir)
        )

        print("📥 Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            "ByteDance/Dolphin",
            cache_dir=str(dolphin_dir)
        )

        print("📥 Downloading model (400MB)...")
        model = AutoModelForVision2Seq.from_pretrained(
            "ByteDance/Dolphin",
            cache_dir=str(dolphin_dir),
            torch_dtype="auto"
        )

        # Save locally
        print("💾 Saving model locally...")
        tokenizer.save_pretrained(dolphin_dir)
        processor.save_pretrained(dolphin_dir)
        model.save_pretrained(dolphin_dir)

        print("✅ Dolphin model downloaded successfully!")
        print(f"📁 Model saved to: {dolphin_dir}")

        # Verify download
        files = list(dolphin_dir.glob("*"))
        print(f"📊 Downloaded {len(files)} files:")
        for file in sorted(files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(".1f")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def test_dolphin_model():
    """Test that the downloaded model works."""

    models_dir = Path("./models")
    dolphin_dir = models_dir / "dolphin"

    if not dolphin_dir.exists():
        print("❌ Dolphin model not found. Run download first.")
        return False

    try:
        print("🧪 Testing Dolphin model...")

        # Load from local directory
        tokenizer = AutoTokenizer.from_pretrained(dolphin_dir)
        processor = AutoProcessor.from_pretrained(dolphin_dir)
        model = AutoModelForVision2Seq.from_pretrained(dolphin_dir)

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