# CUBO - AI Document Assistant

[![CI/CD](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml)

A modular Retrieval-Augmented Generation system using embedding models and Large Language Models (LLMs).

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Multi-format Support**: Supports .txt, .docx, and .pdf files
- **Smart Chunking**: Intelligent text chunking with overlap for better retrieval
- **Device Auto-detection**: Automatically uses GPU (CUDA) if available, falls back to CPU
- **Security Features**: Path sanitization, file size limits, and rate limiting
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Interactive & CLI Modes**: Both interactive conversation and command-line interfaces

## Project Structure

```
cubo/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   ├── model_loader.py    # Model loading and device management
│   ├── document_loader.py # Document loading and processing
│   ├── retriever.py       # Document retrieval logic
│   ├── generator.py       # Response generation
│   ├── utils.py           # Utility functions
│   └── main.py           # Main entry point
├── data/                  # Document storage
├── models/                # Model storage
├── logs/                  # Log files
├── config.json           # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Detailed Installation Guide

### Prerequisites

- **Python 3.8+**: Download from [python.org](https://python.org)
- **Ollama**: Install from [ollama.ai](https://ollama.ai) and pull a model:
  ```bash
  ollama pull llama3.2:latest
  ```
- **Git** (optional): For cloning the repository

### Step-by-Step Installation

1. **Download CUBO**:

   ```bash
   git clone https://github.com/your-repo/cubo.git
   cd cubo
   ```

2. **Create Virtual Environment**:

   ```bash
   python -m venv .venv
   # Activate (Windows)
   .venv\Scripts\activate
   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure**:

   - Edit `config.json` for your settings
   - Place documents in `data/` folder
   - Ensure model files are in `models/` folder

5. **Run Setup Wizard**:
   ```bash
   python src/main.py
   ```
   Follow the prompts to verify paths and models.

### Offline Installation

For air-gapped environments:

- Download all dependencies on an internet-connected machine
- Use `pip download -r requirements.txt -d packages/`
- Transfer to target machine and `pip install --no-index --find-links=packages/ -r requirements.txt`

## Configuration

Edit `config.json` to customize:

- `model_path`: Path to your embedding model
- `llm_model`: Ollama model name (e.g., "llama3.2:latest")
- `top_k`: Number of documents to retrieve
- `chunk_size`: Size of text chunks
- `max_file_size_mb`: Maximum file size limit
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Usage

### Interactive Mode

```bash
python -m src.main
```

### Command Line Mode

```bash
python -m src.main --data_folder ./data --query "Your question here"
```

### From Python

```python
from src.main import CUBOApp

# Initialize app
app = CUBOApp()

# Load documents
app.doc_loader.load_documents_from_folder("./data")

# Add to retriever
app.retriever.add_documents(documents)

# Query
results = app.retriever.retrieve_top_documents("your query")
response = app.generator.generate_response("your query", "\n".join(results))
```

## API Documentation

### Core Classes

#### `CUBOApp`

Main application class handling the full RAG pipeline.

- `setup_wizard()`: Interactive setup for configuration
- `initialize_components()`: Load models and initialize components
- `interactive_mode()`: Run interactive chat interface
- `command_line_mode(args)`: Process single query from CLI

#### `DocumentRetriever`

Handles vector-based document retrieval.

- `__init__(model)`: Initialize with embedding model
- `add_documents(documents)`: Add documents to vector DB
- `retrieve_top_documents(query, top_k=None)`: Retrieve relevant documents

#### `ResponseGenerator`

Generates responses using Ollama LLM.

- `initialize_conversation()`: Start new conversation
- `generate_response(query, context)`: Generate response with context

#### `DocumentLoader`

Loads and processes documents.

- `load_single_document(file_path)`: Load one document
- `load_documents_from_folder(folder_path)`: Load all documents in folder

#### `Utils`

Static utility functions.

- `sanitize_path(path, base_dir)`: Secure path validation
- `validate_file_size(file_path, max_mb)`: Check file size
- `chunk_text(text, chunk_size, overlap)`: Split text into chunks

#### `Config`

Configuration management.

- `get(key, default=None)`: Get config value
- `set(key, value)`: Set config value
- `save()`: Save to file

## Enterprise Use Cases

CUBO is designed for privacy-conscious organizations needing secure, offline AI assistance. Here are key enterprise applications:

### Legal & Compliance

- **Contract Analysis**: Upload legal documents and query for clauses, obligations, or risks
- **Regulatory Compliance**: Search through compliance manuals and policies
- **Due Diligence**: Analyze financial reports and legal filings

### Healthcare & Research

- **Medical Records**: Secure querying of patient data (with proper HIPAA compliance)
- **Research Papers**: Literature review and hypothesis generation
- **Clinical Guidelines**: Access to treatment protocols and drug information

### Technical Documentation

- **Knowledge Base**: Internal documentation search for support teams
- **API Documentation**: Code and API reference lookup
- **Troubleshooting**: System logs and error code analysis

### Education & Training

- **Course Materials**: Student access to lecture notes and textbooks
- **Training Manuals**: Interactive learning with document Q&A
- **Research Assistance**: Academic paper analysis and summarization

### Business Intelligence

- **Report Analysis**: Financial reports, market research, and competitor analysis
- **Policy Documents**: HR policies, procedures, and compliance documents
- **Meeting Notes**: Search through meeting transcripts and action items

### Key Enterprise Benefits

- **Data Privacy**: 100% offline operation - no data leaves your network
- **Cost Effective**: One-time purchase vs. ongoing cloud API costs
- **Customizable**: Modular architecture allows enterprise-specific modifications
- **Scalable**: Handles large document collections with efficient vector search
- **Auditable**: Comprehensive logging for compliance and monitoring

## Requirements

- Python 3.8+
- Ollama (for LLM generation)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Troubleshooting

### TensorFlow Warnings

You may see TensorFlow-related warnings during startup. These are harmless and can be suppressed by setting environment variables:

```bash
# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

On Windows:

```cmd
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
```

### Virtual Environment

Always use a virtual environment to avoid dependency conflicts. The project is designed for isolated installation.

## Security Features

- **Path Sanitization**: Prevents directory traversal attacks
- **File Size Limits**: Configurable maximum file sizes
- **Rate Limiting**: Prevents abuse with configurable delays
- **Input Validation**: Comprehensive input validation

## Logging

Logs are stored in `logs/rag_log.txt` with configurable levels. Check logs for debugging and monitoring.

## Testing

CUBO includes comprehensive unit tests to ensure reliability.

### Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_retriever.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### CI/CD

CUBO uses GitHub Actions for continuous integration:

- **Automated Testing**: Runs on every push/PR to main/master
- **Linting**: Code quality checks with flake8
- **Build**: Automatic exe generation on main branch pushes
- **Artifacts**: Download built exe from Actions tab

Status: ![CI/CD](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml/badge.svg)

### Test Coverage

- **Retriever**: Vector search, caching, similarity thresholds
- **Utils**: Path validation, text processing, file handling
- **Config**: Configuration loading, saving, validation
- **Integration**: End-to-end RAG pipeline testing
