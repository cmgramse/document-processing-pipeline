"""
Main entry point for the document processing pipeline.

Features:
- Document processing with chunking and embedding generation
- Document deletion and cleanup
- Status reporting and statistics
- Error recovery and batch processing

Environment Variables Required:
    JINA_API_KEY: API key for Jina AI
    JINA_EMBEDDING_MODEL: Name of Jina embedding model
    QDRANT_API_KEY: API key for Qdrant
    QDRANT_URL: URL for Qdrant service
    QDRANT_COLLECTION_NAME: Name of Qdrant collection
"""

import os
import logging
import click
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy import inspect

from src.database.session import init_db, get_db
from src.database.models import Document, Chunk
from src.pipeline.processor import (
    process_document,
    process_pending_chunks,
    get_processing_stats
)
from src.processing.documents import chunk_document
from src.database.operations import (
    get_document_status,
    get_unprocessed_files,
    delete_document,
    get_system_stats,
    cleanup_database,
    sync_with_qdrant
)
from src.api.qdrant import validate_qdrant_connection
from src.config import setup_logging

# Configure logging first
setup_logging()
logger = logging.getLogger(__name__)
pipeline_logger = logging.getLogger('pipeline')
error_logger = logging.getLogger('errors')
api_logger = logging.getLogger('api_calls')
metrics_logger = logging.getLogger('metrics')

def validate_environment() -> bool:
    """Validate required environment variables and connections."""
    required_vars = [
        'JINA_API_KEY',
        'JINA_EMBEDDING_MODEL',
        'QDRANT_API_KEY',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            error_logger.error(f"Missing required environment variable: {var}")
            return False
    
    # Validate Qdrant connection
    try:
        if not validate_qdrant_connection():
            raise Exception("Qdrant connection validation failed")
    except Exception as e:
        error_logger.error(f"Failed to connect to Qdrant: {e}")
        return False
    
    return True

def ensure_database() -> bool:
    """Ensure database exists and is properly initialized."""
    try:
        with get_db() as session:
            # Verify schema by checking for required tables
            tables = session.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND 
                name IN ('documents', 'chunks', 'processed_files', 'processing_history')
            """)).fetchall()
            
            if len(tables) < 4:
                logger.warning("Database schema incomplete")
                init_db(drop_all=False)  # Only create missing tables
                logger.info("Database schema restored")
            else:
                # Verify table structure
                inspector = inspect(session.get_bind())
                required_tables = {
                    'documents': {'id', 'filename', 'status'},
                    'chunks': {'id', 'document_id', 'content'},
                    'processed_files': {'filename', 'status'},
                    'processing_history': {'id', 'document_id', 'action'}
                }
                
                for table, required_columns in required_tables.items():
                    columns = {c['name'] for c in inspector.get_columns(table)}
                    if not required_columns.issubset(columns):
                        logger.warning(f"Table {table} missing required columns")
                        return False
            
            return True
            
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        try:
            logger.info("Attempting database recovery...")
            init_db(drop_all=False)  # Try to recover without dropping
            logger.info("Database recovered successfully")
            return True
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False

def ensure_docs_directory() -> Optional[Path]:
    """Ensure documents directory exists and create if missing."""
    docs_path = os.getenv('DOCUMENTS_PATH', 'docs')
    
    # Convert to absolute path if relative
    if not os.path.isabs(docs_path):
        docs_path = os.path.join(os.getcwd(), docs_path)
    
    docs_dir = Path(docs_path)
    try:
        docs_dir.mkdir(exist_ok=True)
        logger.info(f"Documents directory: {docs_dir}")
        return docs_dir
    except Exception as e:
        logger.error(f"Failed to create documents directory: {e}")
        return None

def select_documents(available_docs: List[str], docs_dir: Path) -> List[str]:
    """Interactive document selection."""
    if not available_docs:
        return []
    
    # Show documents with numbers
    print("\nAvailable documents:")
    for i, doc_path in enumerate(available_docs, 1):
        print(f"{i}. {Path(doc_path).relative_to(docs_dir)}")
    
    while True:
        selection = input("\nEnter document numbers (e.g., '1,3'), ranges (e.g., '1-3'), or 'all': ").strip().lower()
        
        if selection == 'all':
            return available_docs
        
        try:
            # Parse selection (handles both individual numbers and ranges)
            selected_indices = set()
            for part in selection.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start-1, end))
                else:
                    selected_indices.add(int(part.strip()) - 1)
            
            # Validate indices
            if not all(0 <= idx < len(available_docs) for idx in selected_indices):
                print("Invalid selection. Please enter valid document numbers.")
                continue
            
            selected_docs = [available_docs[idx] for idx in sorted(selected_indices)]
            
            # Confirm selection
            print("\nSelected documents:")
            for doc in selected_docs:
                print(f"- {Path(doc).relative_to(docs_dir)}")
            
            if input("\nConfirm selection (yes/no): ").lower().startswith('y'):
                return selected_docs
            
        except (ValueError, IndexError):
            print("Invalid input. Please use numbers, ranges, or 'all'")

def check_duplicates(session: Session, doc_paths: List[str]) -> List[str]:
    """Check for duplicate documents and handle user choice."""
    to_process = []
    
    for doc_path in doc_paths:
        filename = Path(doc_path).name
        existing = session.query(Document).filter_by(filename=filename).first()
        
        if existing:
            print(f"\nDocument '{filename}' already exists.")
            choice = input("Options:\n1. Update (removes existing vectors)\n2. Skip\nSelect option (1/2): ")
            
            if choice == '1':
                logger.info(f"Removing existing document '{filename}'...")
                if delete_document(session, filename):
                    to_process.append(doc_path)
                    logger.info("Existing document removed")
                else:
                    logger.error("Failed to remove existing document")
            else:
                logger.info(f"Skipping '{filename}'")
        else:
            to_process.append(doc_path)
    
    return to_process

@click.group()
def cli():
    """Document processing pipeline CLI."""
    if not validate_environment():
        logger.error("Environment validation failed")
        return
    
    # Ensure database exists and is initialized
    if not ensure_database():
        logger.error("Database initialization failed")
        return

@cli.command()
@click.option('--force', is_flag=True, help='Force reprocessing of documents')
def process(force: bool):
    """Process documents and generate embeddings."""
    pipeline_logger.info("Starting document processing pipeline")
    
    # First process any pending chunks
    stats = process_pending_chunks()
    if stats['processed'] > 0:
        pipeline_logger.info(f"Processed {stats['processed']} pending chunks")
    if stats['failed'] > 0:
        error_logger.warning(f"Failed to process {stats['failed']} chunks")
    
    # Ensure documents directory exists
    docs_dir = ensure_docs_directory()
    if not docs_dir:
        error_logger.error("Documents directory not available")
        return
    
    # Get list of documents to process (including markdown files)
    available_docs = []
    for ext in ['*.txt', '*.md']:
        available_docs.extend([
            str(f) for f in docs_dir.glob(f'**/{ext}')
            if not f.name.startswith('.')
        ])
    
    if not available_docs:
        pipeline_logger.info(f"No documents found in {docs_dir}")
        pipeline_logger.info("Please add .txt or .md files to process")
        return
    
    # Get user selection
    selected_docs = select_documents(available_docs, docs_dir)
    if not selected_docs:
        pipeline_logger.info("No documents selected")
        return
    
    # Check for duplicates unless force flag is set
    if not force:
        with get_db() as session:
            selected_docs = check_duplicates(session, selected_docs)
    
    if not selected_docs:
        pipeline_logger.info("No documents to process")
        return
    
    pipeline_logger.info(f"\nProcessing {len(selected_docs)} documents...")
    start_time = datetime.now()
    
    # Process each document
    processed = 0
    failed = 0
    updated = 0
    
    with get_db() as session:
        for doc_path in selected_docs:
            try:
                pipeline_logger.info(f"\nProcessing document: {Path(doc_path).name}")
                
                # Chunk document
                pipeline_logger.info("Chunking document...")
                chunk_texts, token_counts = chunk_document(doc_path)
                pipeline_logger.info(f"Created {len(chunk_texts)} chunks")
                
                # Process document and chunks
                pipeline_logger.info("Saving chunks to database...")
                if process_document(session, doc_path, chunk_texts, token_counts):
                    pipeline_logger.info(f"Successfully processed {Path(doc_path).name}")
                    processed += 1
                else:
                    error_logger.error(f"Failed to process {Path(doc_path).name}")
                    failed += 1
                
            except Exception as e:
                error_logger.error(f"Failed to process {Path(doc_path).name}", exc_info=True)
                failed += 1
                continue
    
    # Process any remaining chunks
    pipeline_logger.info("\nProcessing remaining chunks...")
    chunk_stats = process_pending_chunks()
    
    # Show final statistics
    processing_time = (datetime.now() - start_time).total_seconds()
    
    metrics_logger.info("Processing Complete:")
    metrics_logger.info(f"Documents processed: {processed}")
    metrics_logger.info(f"Documents failed: {failed}")
    metrics_logger.info(f"Documents updated: {updated}")
    metrics_logger.info(f"Chunks total: {chunk_stats['processed'] + chunk_stats['failed']}")
    metrics_logger.info(f"Chunks processed: {chunk_stats['processed']}")
    metrics_logger.info(f"Chunks failed: {chunk_stats['failed']}")
    metrics_logger.info(f"Processing time: {processing_time:.1f} seconds")
    
    # Show user-friendly summary
    print("\nProcessing Complete:")
    print(f"\nDocuments:")
    print(f"- Processed: {processed}")
    print(f"- Failed: {failed}")
    print(f"- Updated: {updated}")
    
    print(f"\nChunks:")
    print(f"- Total: {chunk_stats['processed'] + chunk_stats['failed']}")
    print(f"- Successfully processed: {chunk_stats['processed']}")
    print(f"- Failed: {chunk_stats['failed']}")
    
    print(f"\nProcessing time: {processing_time:.1f} seconds")
    if chunk_stats['processed'] > 0:
        avg_time = processing_time / chunk_stats['processed']
        print(f"Average chunk processing time: {avg_time:.2f} seconds")
        metrics_logger.info(f"Average chunk processing time: {avg_time:.2f} seconds")

@cli.command()
@click.argument('filenames', nargs=-1)
def delete(filenames: List[str]):
    """Delete documents from the system."""
    with get_db() as session:
        if not filenames:
            # Interactive mode
            docs = session.query(Document).all()
            if not docs:
                logger.info("No documents found")
                return
            
            print("\nAvailable documents:")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.filename} (Status: {doc.status})")
            
            selection = input("\nEnter document numbers to delete (comma-separated) or 'all': ")
            if selection.lower() == 'all':
                selected_docs = docs
            else:
                try:
                    indices = [int(i.strip()) - 1 for i in selection.split(',')]
                    selected_docs = [docs[i] for i in indices if 0 <= i < len(docs)]
                except (ValueError, IndexError):
                    logger.error("Invalid selection")
                    return
        else:
            selected_docs = [
                session.query(Document).filter_by(filename=Path(f).name).first()
                for f in filenames
            ]
            selected_docs = [d for d in selected_docs if d]
        
        if not selected_docs:
            logger.info("No documents selected")
            return
        
        print("\nSelected documents for deletion:")
        for doc in selected_docs:
            print(f"- {doc.filename}")
        
        if input("\nConfirm deletion (yes/no): ").lower() != 'yes':
            logger.info("Operation cancelled")
            return
        
        for doc in selected_docs:
            if delete_document(session, doc.filename):
                logger.info(f"Deleted {doc.filename}")
            else:
                logger.error(f"Failed to delete {doc.filename}")

@cli.command()
def stats():
    """Show processing statistics."""
    with get_db() as session:
        processing_stats = get_processing_stats(session)
        system_stats = get_system_stats(session)
        
        print("\nProcessing Statistics:")
        print(f"Documents: {processing_stats['total_documents']}")
        print(f"- Completed: {processing_stats['completed_documents']}")
        print(f"- Failed: {processing_stats['failed_documents']}")
        print(f"- Processing: {processing_stats['processing_documents']}")
        print(f"\nChunks: {processing_stats['total_chunks']}")
        print(f"- Completed: {processing_stats['completed_chunks']}")
        print(f"- Failed: {processing_stats['failed_chunks']}")
        print(f"- Pending: {processing_stats['pending_chunks']}")
        if processing_stats.get('avg_processing_time'):
            print(f"\nAverage processing time: {processing_stats['avg_processing_time']:.2f} seconds")
        
        print("\nSystem Statistics:")
        print(f"Database size: {system_stats['db_size_mb']:.2f} MB")
        print(f"Index size: {system_stats['index_size_mb']:.2f} MB")
        print(f"WAL size: {system_stats['wal_size_mb']:.2f} MB")

@cli.command()
def cleanup():
    """Clean up the database and synchronize with Qdrant."""
    with get_db() as session:
        # Validate Qdrant connection
        try:
            validate_qdrant_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return
        
        # Clean up database
        if cleanup_database(session):
            logger.info("Database cleanup completed")
        else:
            logger.error("Database cleanup failed")
        
        # Sync with Qdrant
        if sync_with_qdrant(session):
            logger.info("Qdrant synchronization completed")
        else:
            logger.error("Qdrant synchronization failed")

if __name__ == '__main__':
    cli()