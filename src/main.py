#!/usr/bin/env python3
"""
CUBO - AI Document Assistant
A portable Retrieval-Augmented Generation system using embedding models and LLMs.
"""

import argparse
import os
import time
from colorama import init
from src.config import config
from src.logger import logger
from src.model_loader import model_manager
from src.document_loader import DocumentLoader
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from src.utils import Utils, metrics
from src.security import security_manager

# Initialize colorama
init()


class CUBOApp:
    """Main application class for CUBO AI Document Assistant."""

    def __init__(self):
        self.model = None
        self.doc_loader = None
        self.retriever = None
        self.generator = None

    def setup_wizard(self):
        """Setup wizard for initial configuration and model checks."""
        logger.info("Welcome to CUBO Setup Wizard!")

        # Security validation
        if not security_manager.validate_environment():
            logger.warning("Some security environment variables are missing.")

        # Check if config.json exists
        try:
            if not os.path.exists("config.json"):
                logger.warning("Config file not found. Creating default config.json...")
                config.save()  # This will create it with defaults
                logger.info("Default config created.")
            else:
                logger.info("Config file found.")
        except Exception as e:
            logger.error(f"Error handling config file: {e}")
            return

        # Check model path
        try:
            model_path = config.get("model_path")
            if not os.path.exists(model_path):
                logger.warning(f"Model path '{model_path}' not found.")
                new_path = input("Enter the correct path to the embedding model (or press Enter to skip): ")
                if new_path:
                    config.set("model_path", new_path)
                    config.save()
                    logger.info("Model path updated.")
                else:
                    logger.warning("Model not found. The application may not work properly.")
            else:
                logger.info("Model path verified.")
        except Exception as e:
            logger.error(f"Error checking model path: {e}")
            return

        # Check data folder
        try:
            data_folder = config.get("data_folder")
            if not os.path.exists(data_folder):
                logger.warning(f"Data folder '{data_folder}' not found. Creating it...")
                os.makedirs(data_folder, exist_ok=True)
                logger.info("Data folder created.")
            else:
                logger.info("Data folder exists.")
        except Exception as e:
            logger.error(f"Error handling data folder: {e}")
            return

        # Check logs folder
        try:
            log_file = config.get("log_file")
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                logger.warning(f"Logs folder '{log_dir}' not found. Creating it...")
                os.makedirs(log_dir, exist_ok=True)
                logger.info("Logs folder created.")
            else:
                logger.info("Logs folder exists.")
        except Exception as e:
            logger.error(f"Error handling logs folder: {e}")
            return

        # LLM Model Selection
        try:
            logger.info("Checking available Ollama models...")
            available_models = self.get_available_ollama_models()
            if available_models:
                logger.info(f"Found {len(available_models)} Ollama models:")
                for i, model in enumerate(available_models, 1):
                    logger.info(f"{i}. {model}")

                current_model = config.get("selected_llm_model", "llama3.2")
                logger.info(f"Current selected model: {current_model}")

                # Auto-select if only one model available
                if len(available_models) == 1:
                    if current_model != available_models[0]:
                        config.set("selected_llm_model", available_models[0])
                        config.save()
                        logger.info(f"Auto-selected only available model: {available_models[0]}")
                    else:
                        logger.info("Only one model available and already selected.")
                else:
                    # Multiple models - let user choose
                    choice = input("Select a model by number (or press Enter to keep current): ")
                    if choice.strip():
                        try:
                            index = int(choice) - 1
                            if 0 <= index < len(available_models):
                                selected_model = available_models[index]
                                config.set("selected_llm_model", selected_model)
                                config.save()
                                logger.info(f"LLM model updated to: {selected_model}")
                            else:
                                logger.warning("Invalid selection. Keeping current model.")
                        except ValueError:
                            logger.warning("Invalid input. Keeping current model.")
                    else:
                        logger.info("Keeping current model.")
            else:
                logger.warning("No Ollama models found. Please install and pull models using 'ollama pull <model_name>'")
                logger.warning("You can change the selected model later in config.json")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")

        # Optional config tweaks
        try:
            tweak = input("Do you want to tweak configuration? (y/n): ").lower()
            if tweak == 'y':
                logger.info("Current config:")
                for key, value in config.all.items():
                    logger.info(f"{key}: {value}")
                key_to_change = input("Enter key to change (or press Enter to skip): ")
                if key_to_change in config.all:
                    new_value = input(f"Enter new value for {key_to_change}: ")
                    config.set(key_to_change, new_value)
                    config.save()
                    logger.info("Config updated.")
        except Exception as e:
            logger.error(f"Error during config tweak: {e}")

    def get_available_ollama_models(self):
        """Get list of available Ollama models."""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is model name
                    return models
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return []

    def initialize_components(self):
        """Initialize model and components."""
        # Load model
        logger.info("Loading embedding model... (this may take a few minutes)")
        start_time = time.time()
        try:
            self.model = model_manager.get_model()
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

        # Initialize components
        self.doc_loader = DocumentLoader()
        self.retriever = DocumentRetriever(self.model)
        self.generator = ResponseGenerator()

        return True

    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        logger.info("Initializing RAG system...")

        # Get data folder
        data_folder_input = input(f"Enter path to data folder "
                                  f"(default: {config.get('data_folder')}): ") or config.get("data_folder")
        data_folder_input = security_manager.sanitize_input(data_folder_input)
        try:
            data_folder = Utils.sanitize_path(data_folder_input, os.getcwd())
        except ValueError as e:
            logger.error(f"Invalid path: {e}")
            return

        if not self.initialize_components():
            return

        # Load documents
        if not os.path.exists(data_folder):
            logger.error(f"Error: Folder '{data_folder}' does not exist.")
            return

        files = [f for f in os.listdir(data_folder) if any(f.endswith(ext) for ext in config.get("supported_extensions"))]
        if not files:
            logger.error(f"No supported files {config.get('supported_extensions')} found in the specified folder.")
            return

        # File selection
        logger.info("Available files:")
        for i, f in enumerate(files, 1):
            logger.info(f"{i}. {f}")

        try:
            choice = int(input("Select file number: ")) - 1
            if choice < 0 or choice >= len(files):
                logger.error("Invalid choice.")
                return
            selected_file = files[choice]
            logger.info("Loading and chunking selected document...")
            start = time.time()
            file_path = os.path.join(data_folder, selected_file)
            documents = self.doc_loader.load_single_document(file_path)
            logger.info(f"Document loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds.")
        except ValueError as e:
            logger.error(f"Error: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return

        logger.info("Documents loaded. Starting conversation. Type 'exit' to quit.")

        # Add to vector DB
        self.retriever.add_documents(documents)

        last_query_time = 0
        while True:
            current_time = time.time()
            if current_time - last_query_time < config.get("rate_limit_seconds", 1):
                time.sleep(config.get("rate_limit_seconds", 1) - (current_time - last_query_time))

            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                logger.info("Exiting conversation.")
                logger.info("Conversation ended.")
                break

            # Retrieve and generate
            top_docs = self.retriever.retrieve_top_documents(query)
            context = "\n".join(top_docs)
            response = self.generator.generate_response(query, context)

            # Audit log the query
            security_manager.audit_log("query_processed", details={"query_hash": security_manager.hash_sensitive_data(query)})

            # Display results
            logger.info("Retrieved Documents:")
            for i, doc in enumerate(top_docs, 1):
                logger.info(f"{i}. {doc[:200]}...")
            logger.info("Response:")
            logger.info(response)

            last_query_time = time.time()
            logger.info(f"Processed query: {query}")

    def command_line_mode(self, args):
        """Run the RAG system in command-line mode."""
        logger.info("Initializing RAG system...")

        try:
            data_folder = Utils.sanitize_path(args.data_folder, os.getcwd())
        except ValueError as e:
            logger.error(f"Invalid path: {e}")
            return

        if not self.initialize_components():
            return

        # Load documents
        logger.info("Loading and chunking all documents...")
        start = time.time()
        documents = self.doc_loader.load_documents_from_folder(data_folder)
        logger.info(f"Documents loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds.")

        # Add to vector DB
        self.retriever.add_documents(documents)

        # Process query
        logger.info("Retrieving top documents...")
        start = time.time()
        top_docs = self.retriever.retrieve_top_documents(args.query)
        logger.info(f"Retrieved in {time.time() - start:.2f} seconds.")

        context = "\n".join(top_docs)
        response = self.generator.generate_response(args.query, context)

        # Output results
        logger.info(f"Query: {args.query}")
        logger.info("Retrieved Documents:")
        for i, doc in enumerate(top_docs, 1):
            logger.info(f"{i}. {doc[:200]}...")
        logger.info("Generated Response:")
        logger.info(response)

    def main(self):
        """Main entry point."""
        start_time = time.time()
        try:
            # Run setup wizard
            self.setup_wizard()

            parser = argparse.ArgumentParser(description="CUBO - AI Document Assistant using embedding model and Llama LLM.")
            parser.add_argument('--data_folder', help="Path to the folder containing documents.")
            parser.add_argument('--query', help="The query to process.")

            args = parser.parse_args()

            if args.data_folder and args.query:
                self.command_line_mode(args)
            else:
                self.interactive_mode()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
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
