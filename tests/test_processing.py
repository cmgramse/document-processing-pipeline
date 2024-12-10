"""
Test script for document processing workflow
"""

import logging
import time
from pathlib import Path
from src.app import DocumentProcessor

def select_document(available_docs):
    """Let user select a single document to process."""
    print("\nAvailable documents:")
    for idx, doc in enumerate(available_docs, 1):
        print(f"{idx}. {doc}")
        
    while True:
        try:
            selection = input("\nEnter the number of the document to process: ")
            idx = int(selection) - 1
            if 0 <= idx < len(available_docs):
                return [available_docs[idx]]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data directory if it doesn't exist
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    processor = DocumentProcessor('./data/documents.db')
    processor.start()
    
    try:
        # List available documents
        from src.processing.documents import list_available_documents
        available_docs = list_available_documents()
        if not available_docs:
            print("No documents found in ./docs directory!")
            return
            
        print(f"\nFound {len(available_docs)} documents")
        
        # Select single document to process
        selected_docs = select_document(available_docs)
        selected_doc = selected_docs[0]
        print(f"\nSelected document: {selected_doc}")
            
        # Process document
        result = processor.process_documents(selected_docs)
        print(f"\nProcessing result:")
        print(f"Selected file: {selected_doc}")
        print(f"Is duplicate: {result['duplicates'].get(selected_doc, False)}")
        print(f"Queued: {result['queued']}")
        
        # Monitor progress
        print("\nMonitoring progress (press Ctrl+C to stop)...")
        while True:
            status = processor.get_processing_status()
            print("\nCurrent status:")
            print(f"Pending: {status['pending']}")
            print(f"Processing: {status['processing']}")
            print(f"Completed: {status['completed']}")
            print(f"Failed: {status['failed']}")
            
            if status['recent_updates']:
                print("\nRecent updates:")
                for update in status['recent_updates'][:5]:
                    print(f"- {update['filename']}: {update['status']}")
                    if update['error']:
                        print(f"  Error: {update['error']}")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\nStopping processor...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main() 