"""
Document upserting script.

This script handles the upserting of documents into the vector database.
"""

import os
import logging
import argparse
from pathlib import Path
import time
from typing import List
import sqlite3
import hashlib

from src.config.settings import ConfigurationManager
from src.processing.document_processing import DocumentProcessor
from src.api.qdrant import QdrantClient
from src.database.init import init_database
from src.database.operations import (
    get_database_stats,
    get_unprocessed_files
)
from src.database.maintenance import optimize_batch_processing
from src.database.connection import DatabaseManager
from src.models.document import Document, ProcessingStatus

def get_available_documents(docs_dir: str = "docs") -> List[str]:
    """Get list of available documents in the docs directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logging.warning(f"Documents directory {docs_dir} does not exist")
        return []
        
    return [
        str(f) for f in docs_path.glob("**/*")
        if f.is_file() and f.suffix in {".md", ".txt", ".rst"}
    ]

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Document upserting script")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--docs-dir", default="docs", help="Documents directory")
    args = parser.parse_args()

    # Load settings
    config = ConfigurationManager()
    settings = config.load()

    # Initialize database
    init_database(settings.db.path)

    # Initialize Qdrant
    qdrant_client = QdrantClient()

    # Get database connection
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_connection() as conn:
            # Initialize processor
            processor = DocumentProcessor()
            
            while True:
                # Get available documents
                available_docs = get_available_documents(args.docs_dir)
                if not available_docs:
                    logging.info("No documents found in docs directory")
                    break
                    
                # Get unprocessed files
                unprocessed_files = get_unprocessed_files(conn=conn, available_docs=available_docs)
                if not unprocessed_files:
                    logging.info("No documents to process")
                    break
                    
                # Convert file paths to Document objects
                documents = []
                for file_path in unprocessed_files:
                    path = Path(file_path)
                    content = path.read_text()
                    doc_id = hashlib.md5(f"{file_path}-{content[:100]}".encode()).hexdigest()
                    
                    documents.append(Document(
                        id=doc_id,
                        title=path.stem,
                        content=content,
                        metadata={"source": file_path},
                        processing_status=ProcessingStatus.PENDING
                    ))
                
                # Process documents
                stats = processor.process_documents(documents=documents)
                if not stats:
                    logging.info("No documents were processed")
                    break
                
                # Wait before next batch
                time.sleep(5)
            
            # Optimize database after processing
            optimize_batch_processing(conn)
            
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 