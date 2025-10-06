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
        self._model_loaded = False

        # Don't load model immediately - load lazily when needed

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

    def _ensure_model_loaded(self):
        """Lazy load the model if not already loaded."""
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process an image with Dolphin model.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (defaults to document parsing)

        Returns:
            Extracted text/content from the image
        """
        # Lazy load model if needed
        self._ensure_model_loaded()

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
        prompt = self._create_extraction_prompt()
        content = self.process_image(image, prompt)

        # Initialize structured data container
        structured = self._initialize_structured_data(content)

        # Parse content with heuristics
        self._parse_content_into_structure(structured, content)

        return structured

    def _create_extraction_prompt(self) -> str:
        """
        Create the prompt for structured data extraction.

        Returns:
            Formatted prompt string for the model
        """
        return (
            "Analyze this document and extract structured information. "
            "Identify and categorize:\n"
            "- Document type/title\n"
            "- Key entities (names, dates, organizations)\n"
            "- Tables (as markdown)\n"
            "- Sections and headers\n"
            "- Important facts or data points\n\n"
            "Format your response as a structured JSON-like output with clear categories."
        )

    def _initialize_structured_data(self, content: str) -> Dict[str, Any]:
        """
        Initialize the structured data container.

        Args:
            content: Raw extracted content from the image

        Returns:
            Dictionary with initial structured data structure
        """
        return {
            "raw_content": content,
            "document_type": "unknown",
            "entities": [],
            "tables": [],
            "sections": []
        }

    def _parse_content_into_structure(self, structured: Dict[str, Any], content: str) -> None:
        """
        Parse content using simple heuristics to populate structured data.

        Args:
            structured: Structured data dictionary to populate
            content: Raw content to parse
        """
        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect headers/sections
            if self._is_header_line(line):
                current_section = line.rstrip(':')
                structured["sections"].append(current_section)
            elif self._is_table_line(line, lines):
                # Likely a table
                structured["tables"].append(line)

    def _is_header_line(self, line: str) -> bool:
        """
        Check if a line appears to be a header or section title.

        Args:
            line: Text line to check

        Returns:
            True if line appears to be a header
        """
        return line.isupper() or (len(line) < 50 and line.endswith(':'))

    def _is_table_line(self, line: str, all_lines: List[str]) -> bool:
        """
        Check if a line appears to be part of a table.

        Args:
            line: Text line to check
            all_lines: All lines in the content for context

        Returns:
            True if line appears to be part of a table
        """
        if '|' not in line:
            return False

        # Check if next line is a separator (common markdown table format)
        try:
            current_index = all_lines.index(line)
            next_index = current_index + 1
            if next_index < len(all_lines):
                next_line = all_lines[next_index].strip()
                return '---' in next_line or ('|' in next_line and '-' in next_line)
        except ValueError:
            pass

        return False

    def is_available(self) -> bool:
        """Check if Dolphin model is available and loaded."""
        try:
            # Trigger lazy loading if not loaded yet
            self._ensure_model_loaded()
            return (
                self.model is not None and
                self.tokenizer is not None and
                self.processor is not None
            )
        except Exception:
            return False
