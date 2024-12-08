"""
Document Management System CLI Application

This script provides a command-line interface for managing a document processing
and storage system. It integrates with Jina AI for text processing and Qdrant
for vector storage.

Features:
- Document processing with chunking and embedding generation
- Document deletion and cleanup
- Status reporting and statistics
- Error recovery and batch processing

Environment Variables Required:
    JINA_API_KEY: API key for Jina AI
    QDRANT_API_KEY: API key for Qdrant
    QDRANT_URL: URL for Qdrant service
    QDRANT_COLLECTION_NAME: Name of Qdrant collection

Example Usage:
    Process documents:
        python main.py process
    
    Delete documents:
        python main.py delete doc1.md doc2.md
    
    View statistics:
        python main.py stats
"""

import os
import sys
import logging
import argparse
from typing import List, Optional
from datetime import datetime

from src.processing.documents import (
    list_available_documents,
    select_documents,
    process_documents
)
from src.api.qdrant import validate_qdrant_connection, initialize_qdrant
from src.database.init import initialize_database
from src.database.maintenance import cleanup_database
from src.management.document_manager import DocumentManager

def setup_logging() -> None:
    """
    Configure logging for the application.
    
    Sets up logging handlers for:
    - Console output (INFO level)
    - File output (DEBUG level)
    - API calls tracking
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log', mode='w')
        ]
    )

def validate_environment() -> bool:
    """
    Validate required environment variables and connections.
    
    Checks:
    - Required API keys are set
    - Database connection is working
    - Qdrant connection is working
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    required_env_vars = ['JINA_API_KEY', 'QDRANT_API_KEY', 'QDRANT_URL', 'QDRANT_COLLECTION_NAME']
    for var in required_env_vars:
        if not os.environ.get(var):
            logging.error(f"Missing required environment variable: {var}")
            return False
    
    # Validate database connection
    try:
        initialize_database()
    except Exception as e:
        logging.error(f"Failed to connect to database: {str(e)}")
        return False
    
    # Validate Qdrant connection
    try:
        validate_qdrant_connection()
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {str(e)}")
        return False
    
    return True

def process_command(args: argparse.Namespace) -> None:
    """
    Handle the process command for document processing.
    
    Args:
        args: Command line arguments
    
    This function:
    1. Lists available documents
    2. Allows document selection
    3. Processes selected documents
    4. Updates processing status
    """
    available_docs = list_available_documents()
    selected_docs = select_documents(available_docs)
    
    if not selected_docs:
        logging.info("No documents selected for processing")
        return
    
    doc_manager = DocumentManager(initialize_database())
    qdrant_client = validate_qdrant_connection()
    
    if not qdrant_client:
        logging.error("Failed to connect to Qdrant")
        return
    
    doc_manager.qdrant_client = qdrant_client
    
    process_documents(selected_docs, doc_manager)

def delete_command(args: argparse.Namespace) -> None:
    """
    Handle the delete command for document removal.
    
    Args:
        args: Command line arguments containing documents to delete
    
    This function:
    1. Validates documents exist
    2. Removes documents from database
    3. Removes vectors from Qdrant
    4. Updates deletion status
    """
    doc_manager = DocumentManager(initialize_database())
    qdrant_client = validate_qdrant_connection()
    
    if not qdrant_client:
        logging.error("Failed to connect to Qdrant")
        return
    
    doc_manager.qdrant_client = qdrant_client
    
    for doc in args.documents:
        if doc_manager.delete_document(doc):
            logging.info(f"Successfully deleted {doc}")
        else:
            logging.error(f"Failed to delete {doc}")

def stats_command(args: argparse.Namespace) -> None:
    """
    Handle the stats command for viewing system statistics.
    
    Args:
        args: Command line arguments
    
    Displays:
    - Total documents processed
    - Processing success rate
    - Storage usage
    - Recent operations
    """
    doc_manager = DocumentManager(initialize_database())
    stats = doc_manager.get_system_stats()
    
    if not stats:
        logging.info("No system statistics available")
        return
    
    for key, value in stats.items():
        logging.info(f"{key}: {value}")

def cleanup_command(args: argparse.Namespace) -> None:
    """
    Handle the cleanup command for system maintenance.
    
    Args:
        args: Command line arguments
    
    This function:
    1. Removes orphaned chunks
    2. Optimizes database
    3. Synchronizes with Qdrant
    """
    doc_manager = DocumentManager(initialize_database())
    qdrant_client = validate_qdrant_connection()
    
    if not qdrant_client:
        logging.error("Failed to connect to Qdrant")
        return
    
    doc_manager.qdrant_client = qdrant_client
    
    if args.dry_run:
        logging.info("Dry run mode, no changes will be made")
    
    cleanup_database(doc_manager, args.dry_run)

def main():
    """
    Main entry point for the document management system.
    
    Handles command line argument parsing and routes to appropriate
    command handlers. Sets up logging and validates environment
    before executing commands.
    
    Commands:
        process: Process new documents
        delete: Remove existing documents
        stats: View system statistics
        cleanup: Perform system maintenance
    """
    parser = argparse.ArgumentParser(
        description='Document Management System'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--force', action='store_true', 
                              help='Force reprocessing of documents')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete documents')
    delete_parser.add_argument('documents', nargs='+', help='Documents to delete')
    
    # Stats command
    subparsers.add_parser('stats', help='View system statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up system')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    setup_logging()
    
    if not validate_environment():
        sys.exit(1)
    
    try:
        if args.command == 'process':
            process_command(args)
        elif args.command == 'delete':
            delete_command(args)
        elif args.command == 'stats':
            stats_command(args)
        elif args.command == 'cleanup':
            cleanup_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error executing command: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()