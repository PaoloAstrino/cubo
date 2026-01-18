#!/usr/bin/env python3
"""
CUBO - AI Document Assistant
A portable Retrieval-Augmented Generation system using embedding models and LLMs.
"""
import sys
import subprocess

# Early short-circuit for lightweight CLI commands so we don't import heavy
# optional dependencies when the user only wants to list models or check
# the version. This keeps the executable responsive and robust in minimal
# environments.
if "--list-models" in sys.argv:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                print("Available models:")
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        print(f" - {parts[0]}")
            else:
                print("No Ollama models found.")
        else:
            print("No Ollama models found or 'ollama' not available.")
    except Exception as e:
        print(f"Could not query Ollama models: {e}")
    sys.exit(0)

if "--version" in sys.argv or "-v" in sys.argv:
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            print(f"CUBO version {version('cubo')}")
        except PackageNotFoundError:
            print("CUBO version 1.0.0")
    except Exception:
        print("CUBO version 1.0.0")
    sys.exit(0)

import argparse
import os
import threading
import time
from typing import Optional

from colorama import init

# Core lightweight imports moved to top level to support static analysis and CI/CD
from cubo.config import config
from cubo.security.security import security_manager
from cubo.utils.logger import logger
from cubo.utils.utils import Utils, metrics
from cubo.ingestion.document_loader import DocumentLoader

# Set global thread control environment variables to reduce OpenMP/BLAS noise
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Defer importing heavy package modules until they're needed by the interactive/setup flows.
# This keeps simple CLI operations (like --list-models or --version) lightweight and
# robust in frozen executables where optional dependencies may be missing or fail to import.

# Initialize colorama
init()


class CUBOApp:
    """Main application class for CUBO AI Document Assistant."""

    def __init__(self):
        self.model = None
        self.doc_loader = None
        self.retriever = None
        self.generator = None
        self.vector_store = None
        # Lock to protect state during build/query operations
        self._state_lock = threading.RLock()

    def _get_version(self) -> str:
        """Return the package version, falling back to a sensible default.

        Tries importlib.metadata.version('cubo') first; if that fails, uses the
        literal fallback '1.0.0'. This ensures the `--version` path is safe and
        does not trigger the setup wizard.
        """
        try:
            from importlib.metadata import version, PackageNotFoundError

            try:
                return version("cubo")
            except PackageNotFoundError:
                return "1.0.0"
        except Exception:
            return "1.0.0"

    def setup_wizard(self, args=None):
        """Setup wizard for initial configuration and model checks.

        Args may contain CLI-driven overrides (like --select-model or --no-interactive)
        to make the flow non-interactive when requested.
        """
        logger.info("Welcome to CUBO Setup Wizard!")

        if not self._validate_security_environment():
            return

        if not self._setup_config_file():
            return

        if not self._validate_model_path():
            return

        if not self._setup_data_folder():
            return

        if not self._setup_logs_folder():
            return

        self._configure_llm_model(args)
        self._optional_config_tweaks(args)


    def _validate_security_environment(self) -> bool:
        """Validate security environment variables."""
        if not security_manager.validate_environment():
            logger.warning("Some security environment variables are missing.")
        return True

    def _setup_config_file(self) -> bool:
        """Ensure config.json exists."""
        try:
            if not os.path.exists("config.json"):
                logger.warning("Config file not found. Creating default config.json...")
                config.save()
                logger.info("Default config created.")
            else:
                logger.info("Config file found.")
            return True
        except Exception as e:
            logger.error(f"Error handling config file: {e}")
            return False

    def _validate_model_path(self) -> bool:
        """Validate and setup model path."""
        try:
            model_path = config.get("model_path")
            if not os.path.exists(model_path):
                logger.warning(f"Model path '{model_path}' not found.")
                new_path = input(
                    "Enter the correct path to the embedding model (or press Enter to skip): "
                )
                if new_path:
                    config.set("model_path", new_path)
                    config.save()
                    logger.info("Model path updated.")
                else:
                    logger.warning("Model not found. The application may not work properly.")
            else:
                logger.info("Model path verified.")
            return True
        except Exception as e:
            logger.error(f"Error checking model path: {e}")
            return False

    def _setup_data_folder(self) -> bool:
        """Ensure data folder exists."""
        try:
            data_folder = config.get("data_folder", "./data")
            if data_folder is None:
                data_folder = "./data"
            if not os.path.exists(data_folder):
                logger.warning(f"Data folder '{data_folder}' not found. Creating it...")
                os.makedirs(data_folder, exist_ok=True)
                logger.info("Data folder created.")
            else:
                logger.info("Data folder exists.")
            return True
        except Exception as e:
            logger.error(f"Error handling data folder: {e}")
            return False

    def _setup_logs_folder(self) -> bool:
        """Ensure logs folder exists."""
        try:
            log_file = config.get("log_file")
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                logger.warning(f"Logs folder '{log_dir}' not found. Creating it...")
                os.makedirs(log_dir, exist_ok=True)
                logger.info("Logs folder created.")
            else:
                logger.info("Logs folder exists.")
            return True
        except Exception as e:
            logger.error(f"Error handling logs folder: {e}")
            return False

    def _configure_llm_model(self, args=None):
        """Configure the LLM model selection and settings."""
        try:
            no_interactive = bool(getattr(args, "no_interactive", False)) if args is not None else False
            logger.info("Checking available Ollama models...")
            available_models = self.get_available_ollama_models(non_interactive=no_interactive)

            # If CLI requested a specific model, apply it non-interactively
            if args is not None and getattr(args, "select_model", None):
                requested = args.select_model
                # If the requested model is among available (base-name match), accept it
                if any(m.split(":")[0] == requested for m in available_models):
                    config.set("selected_llm_model", requested)
                    config.save()
                    print(f"✓ Model set to: {requested}")
                    logger.info(f"CLI requested model '{requested}' applied.")
                    return
                else:
                    # Not available: if non-interactive, still set it (user asked explicitly)
                    if no_interactive:
                        config.set("selected_llm_model", requested)
                        config.save()
                        print(f"✓ Model set to: {requested} (not verified locally)")
                        logger.warning(f"Requested model '{requested}' not found locally; saved in config.")
                        return
                    # Otherwise fall through to interactive selection

            if available_models:
                self._display_available_models(available_models)
                if no_interactive:
                    # Auto-select first model if none configured
                    current_model = config.get("selected_llm_model", None)
                    if current_model is None:
                        selected = available_models[0]
                        config.set("selected_llm_model", selected.split(":")[0])
                        config.save()
                        print(f"✓ Auto-selected model: {selected}")
                        logger.info(f"Auto-selected model: {selected}")
                else:
                    self._handle_model_selection(available_models)
            else:
                logger.warning(
                    "No Ollama models found. Please install and pull models using 'ollama pull <model_name>'"
                )
                logger.warning("You can change the selected model later in config.json")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")


    def _display_available_models(self, available_models):
        """Display available Ollama models."""
        logger.info(f"Found {len(available_models)} Ollama models:")
        for i, model in enumerate(available_models, 1):
            logger.info(f"{i}. {model}")

    def _handle_model_selection(self, available_models):
        """Handle user model selection."""
        current_model = config.get("selected_llm_model", "llama3.2")
        logger.info(f"Current selected model: {current_model}")

        # Skip prompt if current model (base name) is already in available models
        if any(m.split(":")[0] == current_model for m in available_models):
            logger.info(f"Model '{current_model}' is available and configured. Skipping selection.")
            return

        if len(available_models) == 1:
            self._auto_select_single_model(available_models[0], current_model)
        else:
            self._prompt_model_selection(available_models)

    def _auto_select_single_model(self, model, current_model):
        """Auto-select when only one model is available."""
        if current_model != model:
            config.set("selected_llm_model", model)
            config.save()
            logger.info(f"Auto-selected only available model: {model}")
        else:
            logger.info("Only one model available and already selected.")

    def _prompt_model_selection(self, available_models):
        """Prompt user to select from multiple models."""
        print("\n" + "="*50)
        print("Select a model by number (or press Enter to keep current):")
        print("="*50)
        choice = input("> ")
        if choice.strip():
            try:
                index = int(choice) - 1
                if 0 <= index < len(available_models):
                    selected_model = available_models[index]
                    config.set("selected_llm_model", selected_model)
                    config.save()
                    logger.info(f"LLM model updated to: {selected_model}")
                    print(f"✓ Model updated to: {selected_model}")
                else:
                    logger.warning("Invalid selection. Keeping current model.")
            except ValueError:
                logger.warning("Invalid input. Keeping current model.")
        else:
            logger.info("Keeping current model.")

    def _optional_config_tweaks(self):
        """Handle optional configuration tweaks."""
        try:
            tweak = input("Do you want to tweak configuration? (y/n): ").lower()
            if tweak == "y":
                self._display_current_config()
                self._handle_config_change()
        except Exception as e:
            logger.error(f"Error during config tweak: {e}")

    def _display_current_config(self):
        """Display current configuration."""
        logger.info("Current config:")
        for key, value in config.all.items():
            logger.info(f"{key}: {value}")

    def _handle_config_change(self):
        """Handle user config change request."""
        key_to_change = input("Enter key to change (or press Enter to skip): ")
        if key_to_change in config.all:
            new_value = input(f"Enter new value for {key_to_change}: ")
            config.set(key_to_change, new_value)
            config.save()
            logger.info("Config updated.")

    def get_available_ollama_models(self):
        """Get list of available Ollama models in a non-blocking, user-friendly way.

        If the `ollama` CLI is missing or unresponsive, this function prompts the user to
        retry or skip and returns an empty list when models cannot be determined.
        """
        try:
            import subprocess
            import shutil

            # Quick check: is ollama available in PATH?
            if shutil.which("ollama") is None:
                logger.warning("'ollama' binary not found in PATH; skipping model detection.")
                print("Ollama CLI not found in PATH. To use local LLMs install Ollama and run 'ollama pull <model_name>'.")
                print("Press Enter to continue without local LLM or type 'retry' to try again.")
                choice = input("> ").strip().lower()
                if choice == "retry":
                    # Fall through and attempt to run if the user insists
                    pass
                else:
                    return []

            # Try querying ollama with a short timeout and allow the user to retry
            attempts = 0
            while attempts < 3:
                try:
                    print("Checking Ollama models (this may take a few seconds)...")
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                    if result.returncode != 0:
                        logger.warning(f"ollama list failed: {result.stderr.strip()}")
                        print("Failed to query Ollama. Press Enter to continue without local LLM, or type 'retry' to try again.")
                        choice = input("> ").strip().lower()
                        if choice == "retry":
                            attempts += 1
                            continue
                        return []

                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:  # Skip header
                        models = []
                        for line in lines[1:]:
                            parts = line.split()
                            if parts:
                                models.append(parts[0])  # First column is model name
                        return models
                    return []
                except subprocess.TimeoutExpired:
                    logger.warning("Ollama 'list' timed out.")
                    print("Ollama did not respond within 5 seconds. Type 'retry' to try again, or press Enter to continue without local LLM.")
                    choice = input("> ").strip().lower()
                    if choice == "retry":
                        attempts += 1
                        continue
                    return []
        except Exception as e:
            logger.error(f"Unexpected error when checking Ollama models: {e}")
            print(f"Error checking Ollama: {e}. Press Enter to continue without local LLM.")
            input("> ")
            return []

    def initialize_components(self):
        """Initialize model and components."""
        # Load model
        logger.info("Loading embedding model... (this may take a few minutes)")
        start_time = time.time()
        try:
            from cubo.embeddings.model_loader import model_manager
            from cubo.retrieval.retriever import DocumentRetriever
            from cubo.processing.generator import create_response_generator

            with self._state_lock:
                self.model = model_manager.get_model()
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model or components: {e}")
            print(f"✗ Failed to initialize models/components: {e}")
            return False

        # Initialize components - protect state mutation
        with self._state_lock:
            self.doc_loader = DocumentLoader()
            self.retriever = DocumentRetriever(self.model)
            self.generator = create_response_generator()

        return True

    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        logger.info("Initializing RAG system...")

        data_folder = self._get_data_folder_input()
        if not data_folder:
            return

        if not self.initialize_components():
            return

        documents = self._load_selected_document(data_folder)
        if not documents:
            return

        self._run_interactive_conversation(documents)

    def _get_data_folder_input(self) -> str:
        """Get and validate data folder input from user."""
        print("\n" + "="*50)
        print(f"Enter path to data folder (default: {config.get('data_folder')})")
        print("="*50)
        data_folder_input = input("> ") or config.get("data_folder")
        data_folder_input = security_manager.sanitize_input(data_folder_input)
        try:
            return Utils.sanitize_path(data_folder_input, os.getcwd())
        except ValueError as e:
            logger.error(f"Invalid path: {e}")
            print(f"✗ Invalid path: {e}")
            return None

    def _load_selected_document(self, data_folder: str) -> list:
        """Load and validate documents, let user select one."""
        if not os.path.exists(data_folder):
            logger.error(f"Error: Folder '{data_folder}' does not exist.")
            return None

        files = self._get_supported_files(data_folder)
        if not files:
            return None

        selected_file = self._prompt_file_selection(files)
        if not selected_file:
            return None

        return self._load_document_chunks(data_folder, selected_file)

    def _get_supported_files(self, data_folder: str) -> list:
        """Get list of supported files in the data folder."""
        supported_exts = config.get("supported_extensions", [".txt", ".docx", ".pdf", ".md"])
        files = [
            f for f in os.listdir(data_folder) if any(f.endswith(ext) for ext in supported_exts)
        ]
        if not files:
            logger.error(
                f"No supported files {config.get('supported_extensions')} found in the specified folder."
            )
        return files

    def _prompt_file_selection(self, files: list) -> str:
        """Display files and get user selection."""
        print("\n" + "="*50)
        print("Available files:")
        logger.info("Available files:")
        for i, f in enumerate(files, 1):
            print(f"{i}. {f}")
            logger.info(f"{i}. {f}")
        print("="*50)

        try:
            choice = int(input("Select file number: ")) - 1
            if 0 <= choice < len(files):
                return files[choice]
            else:
                logger.error("Invalid choice.")
                print("✗ Invalid choice.")
                return None
        except ValueError:
            logger.error("Invalid input. Please enter a number.")
            print("✗ Invalid input. Please enter a number.")
            return None

    def _load_document_chunks(self, data_folder: str, selected_file: str) -> list:
        """Load and chunk the selected document."""
        try:
            logger.info("Loading and chunking selected document...")
            start = time.time()
            file_path = os.path.join(data_folder, selected_file)
            documents = self.doc_loader.load_single_document(file_path)
            logger.info(
                f"Document loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds."
            )
            return documents
        except Exception as e:
            logger.error(f"Unexpected error loading document: {e}")
            return None

    def _run_interactive_conversation(self, documents: list):
        """Run the interactive conversation loop."""
        logger.info("Documents loaded. Starting conversation. Type 'exit' to quit.")

        # Add to vector DB
        self.retriever.add_documents(documents)

        last_query_time = 0
        while True:
            self._handle_rate_limiting(last_query_time)
            query = self._get_user_query()
            if not query:
                break

            self._process_query(query)
            last_query_time = time.time()

    def _handle_rate_limiting(self, last_query_time: float):
        """Handle rate limiting between queries."""
        current_time = time.time()
        rate_limit = config.get("rate_limit_seconds", 1)
        if current_time - last_query_time < rate_limit:
            time.sleep(rate_limit - (current_time - last_query_time))

    def _get_user_query(self) -> str:
        """Get query input from user."""
        query = input("\nEnter your query: ")
        if query.lower() == "exit":
            logger.info("Exiting conversation.")
            return None
        return query

    def _process_query(self, query: str):
        """Process a single query and display results."""
        # Retrieve and generate
        top_docs = self.retriever.retrieve_top_documents(query)
        context = "\n".join(top_docs)
        response = self.generator.generate_response(query, context)

        # Audit log the query
        security_manager.audit_log(
            "query_processed", details={"query_hash": security_manager.hash_sensitive_data(query)}
        )

        # Display results
        self._display_query_results(top_docs, response, query)

    def _display_query_results(self, top_docs: list, response: str, query: str):
        """Display query results to user."""
        logger.info("Retrieved Documents:")
        for i, doc in enumerate(top_docs, 1):
            logger.info(f"{i}. {doc[:200]}...")
        logger.info("Response:")
        logger.info(response)
        # Respect config 'scrub_queries' to avoid logging raw user queries
        query_to_log = security_manager.scrub(query)
        logger.info(f"Processed query: {query_to_log}")

    def command_line_mode(self, args):
        """Run the RAG system in command-line mode."""
        logger.info("Initializing RAG system...")

        data_folder = self._sanitize_data_folder_path(args)
        if not data_folder:
            return

        if not self.initialize_components():
            return

        documents = self._load_all_documents(data_folder)
        if not documents:
            return

        self._add_documents_to_db(documents)
        self._process_and_display_query(args.query)

    def _sanitize_data_folder_path(self, args) -> str:
        """Sanitize and validate the data folder path."""
        try:
            return Utils.sanitize_path(args.data_folder, os.getcwd())
        except ValueError as e:
            logger.error(f"Invalid path: {e}")
            return None

    def _load_all_documents(self, data_folder: str) -> list:
        """Load and chunk all documents from the data folder."""
        logger.info("Loading and chunking all documents...")
        start = time.time()
        documents = self.doc_loader.load_documents_from_folder(data_folder)
        logger.info(
            f"Documents loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds."
        )
        return documents

    def _add_documents_to_db(self, documents: list):
        """Add documents to the vector database."""
        self.retriever.add_documents(documents)

    # Public API wrappers for programmatic usage (e.g., from API server)
    def ingest_documents(self, data_folder: str = None) -> int:
        """Load and chunk all documents from a folder and return count of chunks.
        This does not add them to the vector DB; call build_index to persist to store.
        """
        folder = data_folder or config.get("data_folder")
        if not self.doc_loader:
            # Ensure doc loader available even if components not fully initialized
            self.doc_loader = DocumentLoader()
        documents = self._load_all_documents(folder)
        return len(documents)

    def build_index(self, data_folder: str = None) -> int:
        """Initialize components if needed, load documents (if any) and add them to the vector DB.
        Returns number of document chunks processed/added.

        Thread-safe: Uses _state_lock to prevent race conditions with queries.
        """
        with self._state_lock:
            # Ensure components are set (model, retriever, generator)
            if not self.model or not self.retriever or not self.generator:
                if not self.initialize_components():
                    raise RuntimeError(
                        "Failed to initialize model and components for index building"
                    )

            folder = data_folder or config.get("data_folder")
            documents = self._load_all_documents(folder)
            if not documents:
                return 0

            # Add documents to the vector DB
            self._add_documents_to_db(documents)
            return len(documents)

    def _process_and_display_query(self, query: str):
        """Process query and display results."""
        logger.info("Retrieving top documents...")
        start = time.time()
        with self._state_lock:
            top_docs = self.retriever.retrieve_top_documents(query)
        logger.info(f"Retrieved in {time.time() - start:.2f} seconds.")

        context = "\n".join(top_docs)
        with self._state_lock:
            response = self.generator.generate_response(query, context)

        # Display results in command line mode (if invoked interactively)
        try:
            self._display_command_line_results(query, top_docs, response)
        except Exception:
            pass

    def query_retrieve(
        self, query: str, top_k: int = None, trace_id: Optional[str] = None, **kwargs
    ):
        """
        Thread-safe wrapper to call the retriever with the state lock.
        """
        with self._state_lock:
            if top_k is None:
                top_k = config.get("retrieval.default_top_k", 6)
            return self.retriever.retrieve_top_documents(query, top_k)

    def generate_response_safe(self, query: str, context: str, trace_id: Optional[str] = None):
        """
        Thread-safe wrapper to call the generator with the state lock.
        """
        with self._state_lock:
            return self.generator.generate_response(query=query, context=context, trace_id=trace_id)

    def _display_command_line_results(self, query: str, top_docs: list, response: str):
        """Display query results in command line format."""
        from cubo.security.security import security_manager

        scrubbed = (
            security_manager.hash_sensitive_data(query)
            if config.get("logging.scrub_queries", False)
            else security_manager.scrub(query)
        )
        logger.info(f"Query: {scrubbed}")
        logger.info("Retrieved Documents:")
        for i, doc in enumerate(top_docs, 1):
            logger.info(f"{i}. {doc[:200]}...")
        logger.info("Generated Response:")
        logger.info(response)

    def main(self):
        """Main entry point."""
        start_time = time.time()
        try:
            # Parse args first to check for quick-exit flags
            args = self._parse_command_line_arguments()

            # If the user only wants to list Ollama models, do that and exit
            if getattr(args, "list_models", False):
                models = self.get_available_ollama_models(non_interactive=getattr(args, "no_interactive", False))
                if models:
                    print("Available models:")
                    for m in models:
                        print(f" - {m}")
                else:
                    print("No Ollama models found or Ollama not available.")
                return

            # Skip setup wizard if requested (or when asking only for version later)
            if not getattr(args, "version", False) and not getattr(args, "skip_setup", False):
                self.setup_wizard(args)

            self._run_application_mode(args)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            self._finalize_application(start_time)

    def _parse_command_line_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="CUBO - AI Document Assistant using embedding model and Llama LLM."
        )
        parser.add_argument("--version", "-v", action="store_true", help="Show version and exit.")
        parser.add_argument("--data_folder", help="Path to the folder containing documents.")
        parser.add_argument("--query", help="The query to process.")

        # Setup & selection options
        parser.add_argument("--skip-setup", action="store_true", help="Skip the interactive setup wizard.")
        parser.add_argument("--select-model", help="Select and save an Ollama model (non-interactive).")
        parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit.")
        parser.add_argument("--no-interactive", action="store_true", help="Run non-interactively and do not prompt the user.")

        # Laptop mode options
        laptop_group = parser.add_mutually_exclusive_group()
        laptop_group.add_argument(
            "--laptop-mode",
            action="store_true",
            help="Force enable laptop mode (reduced resource usage).",
        )
        laptop_group.add_argument(
            "--no-laptop-mode",
            action="store_true",
            help="Disable laptop mode (use full resources).",
        )

        return parser.parse_args()

    def _run_application_mode(self, args):
        """Run the appropriate application mode based on arguments."""
        if getattr(args, "version", False):
            print(f"CUBO version {self._get_version()}")
            return

        # Handle laptop mode flags (overrides auto-detection)
        if getattr(args, "laptop_mode", False):
            config.apply_laptop_mode(force=True)
            logger.info("Laptop mode enabled via --laptop-mode flag.")
        elif getattr(args, "no_laptop_mode", False):
            # Disable laptop mode by resetting relevant settings
            config.set("laptop_mode", False)
            logger.info("Laptop mode disabled via --no-laptop-mode flag.")
        if args.data_folder and args.query:
            self.command_line_mode(args)
        else:
            self.interactive_mode()

    def _finalize_application(self, start_time: float):
        """Finalize application execution with timing and error handling."""
        duration = time.time() - start_time
        metrics.record_time("main_execution", duration)
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")  # Keep terminal open for debugging


if __name__ == "__main__":
    try:
        app = CUBOApp()
        app.main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")  # Keep terminal open for debugging
