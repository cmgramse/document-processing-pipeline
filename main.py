import os
import logging
import argparse
from pathlib import Path
import random
import hashlib
from datetime import datetime

from src.config import setup_logging, check_environment
from src.database.init import initialize_database
from src.database.operations import get_database_stats
from src.database.maintenance import cleanup_database
from src.processing.documents import list_available_documents, select_documents, process_documents
from src.api.qdrant import validate_qdrant_connection, initialize_qdrant
from src.testing.api_tests import test_qdrant_connection, test_jina_apis
from src.management.document_manager import DocumentManager

def main():
    parser = argparse.ArgumentParser(description='Document processing and management system')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and upload documents')
    process_parser.add_argument('--force-reprocess', nargs='*', help='Force reprocess specific files or all if no files specified')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete documents')
    delete_parser.add_argument('files', nargs='*', help='Files to delete')
    delete_parser.add_argument('--force', action='store_true', help='Force delete even if referenced')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old processed chunks')
    cleanup_parser.add_argument('--retention-days', type=int, default=30, help='Number of days to retain processed chunks')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show document statistics')
    stats_parser.add_argument('files', nargs='*', help='Files to show stats for (empty for all)')
    
    args = parser.parse_args()
    
    try:
        # Setup enhanced logging
        api_logger = setup_logging()
        logging.info("Starting document management system")
        
        # Check environment
        check_environment()
        
        # Initialize database connection
        conn = initialize_database()
        
        # Initialize document manager
        doc_manager = DocumentManager(conn)
        
        if args.command == 'cleanup':
            stats = cleanup_database(conn, args.retention_days)
            logging.info(f"Cleanup completed: {stats}")
            return
            
        elif args.command == 'delete':
            if not args.files:
                available_docs = list_available_documents()
                print("\nSelect documents to delete:")
                args.files = doc_manager.batch_process_documents(available_docs)
            
            for file in args.files:
                if doc_manager.delete_document(file, args.force):
                    print(f"Successfully deleted {file}")
                else:
                    print(f"Failed to delete {file}")
            return
            
        elif args.command == 'stats':
            available_docs = list_available_documents()
            if not args.files:
                args.files = available_docs
            
            print("\nDocument Statistics:")
            for file in args.files:
                stats = doc_manager.get_document_stats(file)
                if stats:
                    print(f"\n{file}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
            return
            
        elif args.command == 'process' or not args.command:
            # Validate Qdrant connection
            qdrant_client = validate_qdrant_connection()
            if not qdrant_client:
                raise ConnectionError("Could not connect to Qdrant server")
            
            # Update document manager with Qdrant client
            doc_manager.qdrant_client = qdrant_client
            
            # List and select documents
            available_docs = list_available_documents()
            if not available_docs:
                logging.warning("No documents found in docs directory")
                return
                
            print("\nSelect documents to process:")
            selected_docs = doc_manager.batch_process_documents(available_docs)
            if not selected_docs:
                logging.info("No documents selected for processing")
                return
            
            # Handle existing documents
            to_process = []
            for doc in selected_docs:
                if doc_manager.handle_existing_document(doc):
                    to_process.append(doc)
            
            if not to_process:
                logging.info("No documents to process after handling existing files")
                return
            
            # Process documents
            documents, stats = process_documents(to_process, conn, args.force_reprocess)
            
            if documents:
                # Initialize Qdrant and upload pending documents
                qdrant = initialize_qdrant(conn)
                if qdrant:
                    logging.info("Successfully uploaded pending documents to Qdrant")
                    
                    # Verify random samples
                    num_samples = min(5, len(documents))
                    if num_samples > 0:
                        samples = random.sample(documents, num_samples)
                        test_qdrant_connection(qdrant, samples)
            
            # Show processing statistics
            print("\nProcessing Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()