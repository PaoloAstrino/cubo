import os
from typing import List
from docx import Document
from PyPDF2 import PdfReader
from colorama import Fore, Style
from src.config import config
from src.utils import Utils
from src.logger import logger

class DocumentLoader:
    """Handles loading and processing of various document types for CUBO."""

    def __init__(self):
        self.supported_extensions = config.get("supported_extensions", [".txt", ".docx", ".pdf"])

    def load_single_document(self, file_path: str) -> List[str]:
        """Load and process a single document file."""
        Utils.validate_file_size(file_path, config.get("max_file_size_mb", 10))

        text = ""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_ext == '.docx':
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_ext == '.pdf':
                reader = PdfReader(file_path)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if text:
                text = Utils.clean_text(text)
                chunks = Utils.chunk_text(text)  # Uses adaptive chunking based on text length
                logger.info(f"Loaded and chunked {os.path.basename(file_path)} into {len(chunks)} chunks.")
                return chunks
            else:
                logger.warning(f"No text content found in {file_path}")
                return []

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            print(Fore.YELLOW + f"Warning: Skipping {os.path.basename(file_path)} due to error: {e}" + Style.RESET_ALL)
            return []

    def load_documents_from_folder(self, folder_path: str) -> List[str]:
        """Load all supported documents from a folder."""
        documents = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

        files = [f for f in os.listdir(folder_path)
                if any(f.lower().endswith(ext) for ext in self.supported_extensions)]

        if not files:
            print(Fore.YELLOW + f"No supported files {self.supported_extensions} found in the specified folder." + Style.RESET_ALL)
            return []

        print(Fore.BLUE + f"Loading {len(files)} documents..." + Style.RESET_ALL)

        for file in files:
            file_path = os.path.join(folder_path, file)
            chunks = self.load_single_document(file_path)
            documents.extend(chunks)

        print(Fore.GREEN + f"Total documents loaded and chunked into {len(documents)} chunks." + Style.RESET_ALL)
        return documents
