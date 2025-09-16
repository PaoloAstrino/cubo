#!/usr/bin/env python3
"""
CUBO - AI Document Assistant
A portable Retrieval-Augmented Generation system using embedding models and LLMs.
"""

import argparse
import os
import sys
import time
from colorama import Fore, Style, init
from config import config
from logger import logger
from model_loader import model_manager
from document_loader import DocumentLoader
from retriever import DocumentRetriever
from generator import ResponseGenerator
from utils import Utils

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
        print(Fore.BLUE + "Welcome to CUBO Setup Wizard!" + Style.RESET_ALL)
        
        # Check if config.json exists
        try:
            if not os.path.exists("config.json"):
                print(Fore.YELLOW + "Config file not found. Creating default config.json..." + Style.RESET_ALL)
                config.save()  # This will create it with defaults
                print(Fore.GREEN + "Default config created." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Config file found." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error handling config file: {e}" + Style.RESET_ALL)
            logger.error(f"Config error: {e}")
            return
        
        # Check model path
        try:
            model_path = config.get("model_path")
            if not os.path.exists(model_path):
                print(Fore.YELLOW + f"Model path '{model_path}' not found." + Style.RESET_ALL)
                new_path = input(Fore.YELLOW + "Enter the correct path to the embedding model (or press Enter to skip): " + Style.RESET_ALL)
                if new_path:
                    config.set("model_path", new_path)
                    config.save()
                    print(Fore.GREEN + "Model path updated." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Warning: Model not found. The application may not work properly." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Model path verified." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error checking model path: {e}" + Style.RESET_ALL)
            logger.error(f"Model path error: {e}")
            return
        
        # Check data folder
        try:
            data_folder = config.get("data_folder")
            if not os.path.exists(data_folder):
                print(Fore.YELLOW + f"Data folder '{data_folder}' not found. Creating it..." + Style.RESET_ALL)
                os.makedirs(data_folder, exist_ok=True)
                print(Fore.GREEN + "Data folder created." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Data folder exists." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error handling data folder: {e}" + Style.RESET_ALL)
            logger.error(f"Data folder error: {e}")
            return
        
        # Check logs folder
        try:
            log_file = config.get("log_file")
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                print(Fore.YELLOW + f"Logs folder '{log_dir}' not found. Creating it..." + Style.RESET_ALL)
                os.makedirs(log_dir, exist_ok=True)
                print(Fore.GREEN + "Logs folder created." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Logs folder exists." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error handling logs folder: {e}" + Style.RESET_ALL)
            logger.error(f"Logs folder error: {e}")
            return
        
        # Optional config tweaks
        try:
            tweak = input(Fore.YELLOW + "Do you want to tweak configuration? (y/n): " + Style.RESET_ALL).lower()
            if tweak == 'y':
                print(Fore.CYAN + "Current config:" + Style.RESET_ALL)
                for key, value in config.all.items():
                    print(f"{key}: {value}")
                key_to_change = input(Fore.YELLOW + "Enter key to change (or press Enter to skip): " + Style.RESET_ALL)
                if key_to_change in config.all:
                    new_value = input(Fore.YELLOW + f"Enter new value for {key_to_change}: " + Style.RESET_ALL)
                    config.set(key_to_change, new_value)
                    config.save()
                    print(Fore.GREEN + "Config updated." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error during config tweak: {e}" + Style.RESET_ALL)
            logger.error(f"Config tweak error: {e}")

    def initialize_components(self):
        """Initialize model and components."""
        # Load model
        print(Fore.BLUE + "Loading embedding model... (this may take a few minutes)" + Style.RESET_ALL)
        start_time = time.time()
        try:
            self.model = model_manager.get_model()
            print(Fore.GREEN + f"Model loaded successfully in {time.time() - start_time:.2f} seconds." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Failed to load model: {e}" + Style.RESET_ALL)
            logger.error(f"Model loading error: {e}")
            return False

        # Initialize components
        self.doc_loader = DocumentLoader()
        self.retriever = DocumentRetriever(self.model)
        self.generator = ResponseGenerator()
        self.generator.initialize_conversation()
        return True

    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        print(Fore.BLUE + "Initializing RAG system..." + Style.RESET_ALL)

        # Get data folder
        data_folder_input = input(Fore.YELLOW + f"Enter path to data folder (default: {config.get('data_folder')}): " + Style.RESET_ALL) or config.get("data_folder")
        try:
            data_folder = Utils.sanitize_path(data_folder_input, os.getcwd())
        except ValueError as e:
            print(Fore.RED + f"Invalid path: {e}" + Style.RESET_ALL)
            logger.error(f"Invalid path: {e}")
            return

        if not self.initialize_components():
            return

        # Load documents
        if not os.path.exists(data_folder):
            print(Fore.RED + f"Error: Folder '{data_folder}' does not exist." + Style.RESET_ALL)
            return

        files = [f for f in os.listdir(data_folder) if any(f.endswith(ext) for ext in config.get("supported_extensions"))]
        if not files:
            print(Fore.RED + f"No supported files {config.get('supported_extensions')} found in the specified folder." + Style.RESET_ALL)
            return

        # File selection
        print(Fore.CYAN + "Available files:" + Style.RESET_ALL)
        for i, f in enumerate(files, 1):
            print(f"{i}. {f}")

        try:
            choice = int(input(Fore.YELLOW + "Select file number: " + Style.RESET_ALL)) - 1
            if choice < 0 or choice >= len(files):
                print(Fore.RED + "Invalid choice." + Style.RESET_ALL)
                return
            selected_file = files[choice]
            print(Fore.BLUE + "Loading and chunking selected document..." + Style.RESET_ALL)
            start = time.time()
            file_path = os.path.join(data_folder, selected_file)
            documents = self.doc_loader.load_single_document(file_path)
            print(Fore.GREEN + f"Document loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
        except ValueError as e:
            print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
            return
        except Exception as e:
            print(Fore.RED + f"Unexpected error: {e}" + Style.RESET_ALL)
            return

        print(Fore.GREEN + "Documents loaded. Starting conversation. Type 'exit' to quit." + Style.RESET_ALL)

        # Add to vector DB
        self.retriever.add_documents(documents)

        last_query_time = 0
        while True:
            current_time = time.time()
            if current_time - last_query_time < config.get("rate_limit_seconds", 1):
                time.sleep(config.get("rate_limit_seconds", 1) - (current_time - last_query_time))

            query = input(Fore.YELLOW + "\nEnter your query: " + Style.RESET_ALL)
            if query.lower() == 'exit':
                print(Fore.BLUE + "Exiting conversation." + Style.RESET_ALL)
                logger.info("Conversation ended.")
                break

            # Retrieve and generate
            top_docs = self.retriever.retrieve_top_documents(query)
            context = "\n".join(top_docs)
            response = self.generator.generate_response(query, context)

            # Display results
            print(Fore.CYAN + "Retrieved Documents:" + Style.RESET_ALL)
            for i, doc in enumerate(top_docs, 1):
                print(f"{i}. {doc[:200]}...")
            print(Fore.CYAN + "Response:" + Style.RESET_ALL)
            print(response)

            last_query_time = time.time()
            logger.info(f"Processed query: {query}")

    def command_line_mode(self, args):
        """Run the RAG system in command-line mode."""
        print(Fore.BLUE + "Initializing RAG system..." + Style.RESET_ALL)

        try:
            data_folder = Utils.sanitize_path(args.data_folder, os.getcwd())
        except ValueError as e:
            print(Fore.RED + f"Invalid path: {e}" + Style.RESET_ALL)
            return

        if not self.initialize_components():
            return

        # Load documents
        print(Fore.BLUE + "Loading and chunking all documents..." + Style.RESET_ALL)
        start = time.time()
        documents = self.doc_loader.load_documents_from_folder(data_folder)
        print(Fore.GREEN + f"Documents loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds." + Style.RESET_ALL)

        # Add to vector DB
        self.retriever.add_documents(documents)

        # Process query
        print(Fore.BLUE + "Retrieving top documents..." + Style.RESET_ALL)
        start = time.time()
        top_docs = self.retriever.retrieve_top_documents(args.query)
        print(Fore.GREEN + f"Retrieved in {time.time() - start:.2f} seconds." + Style.RESET_ALL)

        context = "\n".join(top_docs)
        response = self.generator.generate_response(args.query, context)

        # Output results
        print(Fore.CYAN + "Query:" + Style.RESET_ALL, args.query)
        print(Fore.CYAN + "Retrieved Documents:" + Style.RESET_ALL)
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. {doc[:200]}...")
        print(Fore.CYAN + "Generated Response:" + Style.RESET_ALL)
        print(response)

    def main(self):
        """Main entry point."""
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
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc()
            input("Press Enter to exit...")  # Keep terminal open for debugging

if __name__ == "__main__":
    try:
        app = CUBOApp()
        app.main()
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # Keep terminal open for debugging
