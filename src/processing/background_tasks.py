"""
Background Tasks Module

This module provides background processing capabilities for the document processing system.
It handles tasks that can run asynchronously, such as embedding generation for chunks.
"""

import logging
import time
import threading
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
from ..database.operations import process_pending_chunks, get_unprocessed_files
from ..api.jina import jina_client, process_document
from ..api.qdrant import upsert_embeddings
from src.monitoring.metrics import log_document_processing

class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"  # Combined chunking and embedding
    UPSERTING = "upserting"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingQueue:
    """Queue for managing document processing."""
    def __init__(self):
        self.queue = {}  # filename -> status
        self.errors = {}  # filename -> error message
        self._lock = threading.Lock()

    def add_documents(self, filenames: List[str]) -> None:
        """Add documents to the processing queue."""
        with self._lock:
            for filename in filenames:
                if filename not in self.queue:
                    self.queue[filename] = ProcessingStatus.PENDING
                    logging.info(f"Added {filename} to processing queue")

    def get_next_pending(self) -> Optional[Dict[str, Any]]:
        """Get the next pending document."""
        with self._lock:
            for filename, status in self.queue.items():
                if status == ProcessingStatus.PENDING:
                    self.queue[filename] = ProcessingStatus.PROCESSING
                    return {"filename": filename}
            return None

    def update_status(self, filename: str, status: ProcessingStatus, error: str = None) -> None:
        """Update document processing status."""
        with self._lock:
            self.queue[filename] = status
            if error:
                self.errors[filename] = error
            logging.info(f"Updated {filename} status to {status.value}")

    def get_status(self, filename: str) -> Tuple[ProcessingStatus, Optional[str]]:
        """Get document status and error if any."""
        with self._lock:
            return self.queue.get(filename, ProcessingStatus.PENDING), self.errors.get(filename)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all documents."""
        with self._lock:
            return {
                filename: {
                    "status": status.value,
                    "error": self.errors.get(filename)
                }
                for filename, status in self.queue.items()
            }

    def clear_completed(self) -> None:
        """Remove completed documents from the queue."""
        with self._lock:
            completed = [f for f, s in self.queue.items() if s == ProcessingStatus.COMPLETED]
            for filename in completed:
                del self.queue[filename]
                if filename in self.errors:
                    del self.errors[filename]
            if completed:
                logging.info(f"Cleared {len(completed)} completed documents from queue")

class BackgroundProcessor:
    """Background processor for handling asynchronous tasks."""
    
    def __init__(self, db_path: str, batch_size: int = 10, check_interval: int = 30):
        """
        Initialize the background processor.
        
        Args:
            db_path: Path to the SQLite database
            batch_size: Number of documents to process in each batch
            check_interval: Seconds between checks for pending documents
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.check_interval = check_interval
        self.should_run = False
        self.thread: Optional[threading.Thread] = None
        self.queue = ProcessingQueue()
        
    def start(self):
        """Start the background processor."""
        if self.thread is not None and self.thread.is_alive():
            logging.warning("Background processor is already running")
            return
            
        self.should_run = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logging.info("Background processor started")
        
    def stop(self):
        """Stop the background processor."""
        self.should_run = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        logging.info("Background processor stopped")
        
    def _process_document(self, doc: Dict[str, Any]):
        """Process a single document through all stages."""
        try:
            filename = doc["filename"]
            file_path = Path('./docs') / filename
            logging.info(f"Processing document: {filename}")
            
            if not file_path.exists():
                raise Exception(f"File not found: {file_path}")
            
            # Process document using consolidated Jina API
            logging.info(f"Starting processing for {filename}")
            self.queue.update_status(filename, ProcessingStatus.PROCESSING)
            
            processed_chunks = process_document(
                file_path.read_text(),
                source=filename
            )
            
            if not processed_chunks:
                raise Exception("Processing failed - no chunks generated")
            
            logging.info(f"Generated {len(processed_chunks)} chunks for {filename}")
            
            # Convert to Document objects for Qdrant
            documents = []
            embeddings = []
            for chunk in processed_chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata=chunk['metadata']
                )
                documents.append(doc)
                embeddings.append(chunk['embedding'])
            
            # Upserting stage
            logging.info(f"Starting upserting stage for {filename}")
            self.queue.update_status(filename, ProcessingStatus.UPSERTING)
            if not upsert_embeddings(documents, embeddings):
                raise Exception("Failed to upsert embeddings")
            logging.info(f"Completed upserting for {filename}")
            
            # Mark as completed
            self.queue.update_status(filename, ProcessingStatus.COMPLETED)
            logging.info(f"Successfully processed {filename}")
            
            processing_time = time.time() - start_time
            log_document_processing(
                success=True,
                chunks_created=len(processed_chunks),
                embeddings_generated=len(processed_chunks),
                processing_time=processing_time,
                details={
                    'filename': filename,
                    'content_length': len(file_path.read_text())
                }
            )
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            self.queue.update_status(filename, ProcessingStatus.FAILED, str(e))
            log_document_processing(
                success=False,
                details={
                    'filename': filename,
                    'error': str(e)
                }
            )
        
    def _run(self):
        """Main processing loop."""
        while self.should_run:
            try:
                logging.info("Checking for documents to process...")
                batch = self.queue.get_next_pending()
                if batch:
                    logging.info(f"Found {len(batch)} documents to process")
                    for doc in batch:
                        self._process_document(doc)
                else:
                    logging.info("No documents found to process")
            except Exception as e:
                logging.error(f"Error in background processor: {str(e)}")
                
            time.sleep(self.check_interval)