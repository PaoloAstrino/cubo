#!/usr/bin/env python3
"""
Dolphin Processor for CUBO
Integrates ByteDance/Dolphin vision-language model with EmbeddingGemma-300M
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)


class DolphinProcessor:
    """Wrapper for ByteDance/Dolphin vision-language model."""

    def __init__(self, model_path: str = "./models/dolphin"):
        """
        Initialize Dolphin processor.

        Args:
            model_path: Path to the Dolphin model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model()

    def _load_model(self):
        """Load the Dolphin model from local directory."""
        try:
            logger.info(f"Loading Dolphin model from {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

            logger.info("Dolphin model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Dolphin model: {e}")
            raise

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process an image with Dolphin model.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (defaults to document parsing)

        Returns:
            Extracted text/content from the image
        """
        if prompt is None:
            prompt = (
                "You are an expert document analyzer. Extract all text, tables, "
                "and structured information from this document image. "
                "Maintain the original formatting and structure as much as possible. "
                "If there are tables, represent them in markdown format. "
                "Be thorough and accurate."
            )

        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1
                )

            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return ""

    def process_document_pages(self, images: List[Image.Image],
                               page_prompts: Optional[List[str]] = None) -> List[str]:
        """
        Process multiple document pages.

        Args:
            images: List of PIL Images (one per page)
            page_prompts: Optional custom prompts per page

        Returns:
            List of extracted content per page
        """
        results = []

        for i, image in enumerate(images):
            prompt = page_prompts[i] if page_prompts and i < len(page_prompts) else None
            logger.info(f"Processing page {i+1}/{len(images)}")

            content = self.process_image(image, prompt)
            results.append(content)

        return results

    def extract_structured_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract structured data from document image.

        Args:
            image: PIL Image of the document

        Returns:
            Dictionary with extracted structured information
        """
        prompt = (
            "Analyze this document and extract structured information. "
            "Identify and categorize:\n"
            "- Document type/title\n"
            "- Key entities (names, dates, organizations)\n"
            "- Tables (as markdown)\n"
            "- Sections and headers\n"
            "- Important facts or data points\n\n"
            "Format your response as a structured JSON-like output with clear categories."
        )

        content = self.process_image(image, prompt)

        # Basic parsing (could be enhanced with better NLP)
        structured = {
            "raw_content": content,
            "document_type": "unknown",
            "entities": [],
            "tables": [],
            "sections": []
        }

        # Simple heuristics for structure detection
        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect headers/sections
            if line.isupper() or (len(line) < 50 and line.endswith(':')):
                current_section = line.rstrip(':')
                structured["sections"].append(current_section)
            elif '|' in line and ('---' in lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else False):
                # Likely a table
                structured["tables"].append(line)

        return structured

    def is_available(self) -> bool:
        """Check if Dolphin model is available and loaded."""
        return (
            self.model is not None and
            self.tokenizer is not None and
            self.processor is not None
        )
