#!/usr/bin/env python3
"""
RAG System - Main Entry Point
A portable Retrieval-Augmented Generation system using embedding models and LLMs.
"""

import argparse
import os
import sys
import time
from colorama import Fore, Style, init
from .config import config
from .logger import setup_logging
from .model_loader import model_manager
from .document_loader import DocumentLoader
from .retriever import DocumentRetriever
from .generator import ResponseGenerator
from .utils import sanitize_path

# Initialize colorama
init()

# Setup logging
logger = setup_logging()

def interactive_mode():
    """Run the RAG system in interactive mode."""
    print(Fore.BLUE + "Initializing RAG system..." + Style.RESET_ALL)

    # Get data folder
    data_folder_input = input(Fore.YELLOW + f"Enter path to data folder (default: {config.get('data_folder')}): " + Style.RESET_ALL) or config.get("data_folder")
    try:
        data_folder = sanitize_path(data_folder_input, os.getcwd())
    except ValueError as e:
        print(Fore.RED + f"Invalid path: {e}" + Style.RESET_ALL)
        logger.error(f"Invalid path: {e}")
        return

    # Load model
    try:
        model = model_manager.get_model()
    except Exception as e:
        return

    # Initialize components
    doc_loader = DocumentLoader()
    retriever = DocumentRetriever(model)
    generator = ResponseGenerator()
    generator.initialize_conversation()

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
        documents = doc_loader.load_single_document(file_path)
        print(Fore.GREEN + f"Document loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
    except ValueError as e:
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        return
    except Exception as e:
        print(Fore.RED + f"Unexpected error: {e}" + Style.RESET_ALL)
        return

    print(Fore.GREEN + "Documents loaded. Starting conversation. Type 'exit' to quit." + Style.RESET_ALL)

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
        top_docs = retriever.retrieve_top_documents(query, documents)
        context = "\n".join(top_docs)
        response = generator.generate_response(query, context)

        # Display results
        print(Fore.CYAN + "Retrieved Documents:" + Style.RESET_ALL)
        for i, doc in enumerate(top_docs, 1):
            print(f"{i}. {doc[:200]}...")
        print(Fore.CYAN + "Response:" + Style.RESET_ALL)
        print(response)

        last_query_time = time.time()
        logger.info(f"Processed query: {query}")

def command_line_mode(args):
    """Run the RAG system in command-line mode."""
    print(Fore.BLUE + "Initializing RAG system..." + Style.RESET_ALL)

    try:
        data_folder = sanitize_path(args.data_folder, os.getcwd())
    except ValueError as e:
        print(Fore.RED + f"Invalid path: {e}" + Style.RESET_ALL)
        return

    # Load model
    try:
        model = model_manager.get_model()
    except Exception as e:
        return

    # Initialize components
    doc_loader = DocumentLoader()
    retriever = DocumentRetriever(model)
    generator = ResponseGenerator()
    generator.initialize_conversation()

    # Load documents
    print(Fore.BLUE + "Loading and chunking all documents..." + Style.RESET_ALL)
    start = time.time()
    documents = doc_loader.load_documents_from_folder(data_folder)
    print(Fore.GREEN + f"Documents loaded and chunked into {len(documents)} chunks in {time.time() - start:.2f} seconds." + Style.RESET_ALL)

    # Process query
    print(Fore.BLUE + "Retrieving top documents..." + Style.RESET_ALL)
    start = time.time()
    top_docs = retriever.retrieve_top_documents(args.query, documents)
    print(Fore.GREEN + f"Retrieved in {time.time() - start:.2f} seconds." + Style.RESET_ALL)

    context = "\n".join(top_docs)
    response = generator.generate_response(args.query, context)

    # Output results
    print(Fore.CYAN + "Query:" + Style.RESET_ALL, args.query)
    print(Fore.CYAN + "Retrieved Documents:" + Style.RESET_ALL)
    for i, doc in enumerate(top_docs, 1):
        print(f"{i}. {doc[:200]}...")
    print(Fore.CYAN + "Generated Response:" + Style.RESET_ALL)
    print(response)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG system using embedding model and Llama LLM.")
    parser.add_argument('--data_folder', help="Path to the folder containing documents.")
    parser.add_argument('--query', help="The query to process.")

    args = parser.parse_args()

    if args.data_folder and args.query:
        command_line_mode(args)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
