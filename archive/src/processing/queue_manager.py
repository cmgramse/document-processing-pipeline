"""
Queue Management Module

Provides unified queue management for document processing.
"""

import logging
import threading
import asyncio
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from pathlib import Path

from ..models.document import Document, ProcessingStatus
from ..database.document_manager import DocumentManager
from .resource_monitor import ResourceMonitor
from .background_tasks import process_document_async
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass(order=True)
class QueueItem:
    """Item in the processing queue."""
    priority: int
    document_id: str = field(compare=False)
    added_at: datetime = field(default_factory=datetime.now, compare=False)
    started_at: Optional[datetime] = field(default=None, compare=False)
    completed_at: Optional[datetime] = field(default=None, compare=False)
    status: ProcessingStatus = field(default=ProcessingStatus.PENDING, compare=False)
    error_message: Optional[str] = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)
    next_retry_at: Optional[datetime] = field(default=None, compare=False)

class QueueManager:
    """
    Manages document processing queue.
    Implements singleton pattern to ensure single queue across application.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize queue manager if not already initialized."""
        if not self._initialized:
            self._queue: PriorityQueue = PriorityQueue()
            self._items: Dict[str, QueueItem] = {}
            self._processing: Dict[str, QueueItem] = {}
            self._completed: Dict[str, QueueItem] = {}
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._worker_thread = None
            self._retry_thread = None
            self._document_manager = DocumentManager()
            self._resource_monitor = ResourceMonitor()
            self._db_path = Path(settings.db_path).parent / "queue.db"
            self._init_db()
            self._restore_state()
            self._initialized = True
    
    def _init_db(self) -> None:
        """Initialize queue database."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            c = conn.cursor()
            
            # Queue items table
            c.execute("""
            CREATE TABLE IF NOT EXISTS queue_items (
                document_id TEXT PRIMARY KEY,
                priority INTEGER NOT NULL,
                added_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT NOT NULL,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_at TIMESTAMP
            )
            """)
            
            # Create indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON queue_items(status)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_queue_retry ON queue_items(next_retry_at)")
            
            conn.commit()
        finally:
            conn.close()
    
    def _restore_state(self) -> None:
        """Restore queue state from database."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM queue_items")
            
            for row in c.fetchall():
                item = QueueItem(
                    priority=row[1],
                    document_id=row[0],
                    added_at=datetime.fromisoformat(row[2]),
                    started_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    status=ProcessingStatus(row[5]),
                    error_message=row[6],
                    retry_count=row[7],
                    next_retry_at=datetime.fromisoformat(row[8]) if row[8] else None
                )
                
                if item.status == ProcessingStatus.PENDING:
                    self._items[item.document_id] = item
                    self._queue.put(item)
                elif item.status == ProcessingStatus.PROCESSING:
                    self._processing[item.document_id] = item
                else:
                    self._completed[item.document_id] = item
        finally:
            conn.close()
    
    def _save_item(self, item: QueueItem) -> None:
        """Save queue item to database."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT OR REPLACE INTO queue_items (
                    document_id, priority, added_at, started_at,
                    completed_at, status, error_message,
                    retry_count, next_retry_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.document_id,
                    item.priority,
                    item.added_at.isoformat(),
                    item.started_at.isoformat() if item.started_at else None,
                    item.completed_at.isoformat() if item.completed_at else None,
                    item.status.value,
                    item.error_message,
                    item.retry_count,
                    item.next_retry_at.isoformat() if item.next_retry_at else None
                )
            )
            conn.commit()
        finally:
            conn.close()
    
    def start(self) -> None:
        """Start queue processing."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            
            # Start main worker thread
            self._worker_thread = threading.Thread(target=self._process_queue)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            
            # Start retry handler thread
            self._retry_thread = threading.Thread(target=self._handle_retries)
            self._retry_thread.daemon = True
            self._retry_thread.start()
            
            logger.info("Queue processing started")
    
    def stop(self) -> None:
        """Stop queue processing."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            self._worker_thread.join()
            if self._retry_thread:
                self._retry_thread.join()
            logger.info("Queue processing stopped")
    
    def add_document(self, document: Document, priority: int = 1) -> None:
        """
        Add document to processing queue.
        
        Args:
            document: Document to process
            priority: Processing priority (lower number = higher priority)
        """
        with self._lock:
            if document.id not in self._items:
                item = QueueItem(priority=priority, document_id=document.id)
                self._items[document.id] = item
                self._queue.put(item)
                self._save_item(item)
                logger.info(f"Document {document.id} added to queue (priority: {priority})")
    
    def add_documents(self, documents: List[Document], priority: int = 1) -> None:
        """Add multiple documents to processing queue."""
        for document in documents:
            self.add_document(document, priority)
    
    def get_status(self, document_id: str) -> 'Tuple[ProcessingStatus, Optional[str]]':
        """Get processing status of a document."""
        with self._lock:
            if document_id in self._items:
                item = self._items[document_id]
                return item.status, item.error_message
            if document_id in self._processing:
                item = self._processing[document_id]
                return item.status, item.error_message
            if document_id in self._completed:
                item = self._completed[document_id]
                return item.status, item.error_message
        return ProcessingStatus.PENDING, None
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all documents in queue."""
        with self._lock:
            statuses = {}
            
            # Add queued items
            for doc_id, item in self._items.items():
                statuses[doc_id] = {
                    'status': item.status.value,
                    'error': item.error_message,
                    'queued_at': item.added_at.isoformat(),
                    'started_at': None,
                    'completed_at': None,
                    'retry_count': item.retry_count,
                    'next_retry_at': item.next_retry_at.isoformat() if item.next_retry_at else None,
                    'priority': item.priority
                }
            
            # Add processing items
            for doc_id, item in self._processing.items():
                statuses[doc_id] = {
                    'status': item.status.value,
                    'error': item.error_message,
                    'queued_at': item.added_at.isoformat(),
                    'started_at': item.started_at.isoformat() if item.started_at else None,
                    'completed_at': None,
                    'retry_count': item.retry_count,
                    'next_retry_at': item.next_retry_at.isoformat() if item.next_retry_at else None,
                    'priority': item.priority
                }
            
            # Add completed items
            for doc_id, item in self._completed.items():
                statuses[doc_id] = {
                    'status': item.status.value,
                    'error': item.error_message,
                    'queued_at': item.added_at.isoformat(),
                    'started_at': item.started_at.isoformat() if item.started_at else None,
                    'completed_at': item.completed_at.isoformat() if item.completed_at else None,
                    'retry_count': item.retry_count,
                    'next_retry_at': None,
                    'priority': item.priority
                }
            
            return statuses
    
    def clear_completed(self) -> None:
        """Clear completed items from queue."""
        with self._lock:
            # Remove from database
            conn = sqlite3.connect(str(self._db_path))
            try:
                c = conn.cursor()
                c.execute(
                    "DELETE FROM queue_items WHERE status IN (?, ?)",
                    (ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value)
                )
                conn.commit()
            finally:
                conn.close()
            
            # Clear from memory
            self._completed.clear()
    
    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate delay before next retry in seconds."""
        # Exponential backoff with jitter
        import random
        base_delay = min(300, 30 * (2 ** retry_count))  # Cap at 5 minutes
        jitter = random.uniform(0.8, 1.2)
        return int(base_delay * jitter)
    
    def _handle_retries(self) -> None:
        """Handle retrying failed items."""
        while not self._stop_event.is_set():
            try:
                conn = sqlite3.connect(str(self._db_path))
                c = conn.cursor()
                
                # Find items ready for retry
                c.execute(
                    """
                    SELECT document_id FROM queue_items
                    WHERE status = ? AND next_retry_at <= datetime('now')
                    """,
                    (ProcessingStatus.FAILED.value,)
                )
                
                for (doc_id,) in c.fetchall():
                    with self._lock:
                        if doc_id in self._completed:
                            item = self._completed.pop(doc_id)
                            item.status = ProcessingStatus.PENDING
                            item.retry_count += 1
                            item.next_retry_at = None
                            self._items[doc_id] = item
                            self._queue.put(item)
                            self._save_item(item)
                            logger.info(
                                f"Retrying document {doc_id} "
                                f"(attempt {item.retry_count})"
                            )
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error in retry handler: {e}")
            
            # Check every minute
            self._stop_event.wait(60)
    
    async def _process_document(self, item: QueueItem) -> None:
        """Process a single document."""
        try:
            await process_document_async(item.document_id)
            
            with self._lock:
                if item.document_id in self._processing:
                    item = self._processing.pop(item.document_id)
                    item.status = ProcessingStatus.COMPLETED
                    item.completed_at = datetime.now()
                    self._completed[item.document_id] = item
                    self._save_item(item)
                    
        except Exception as e:
            logger.error(f"Error processing document {item.document_id}: {e}")
            with self._lock:
                if item.document_id in self._processing:
                    item = self._processing.pop(item.document_id)
                    item.status = ProcessingStatus.FAILED
                    item.error_message = str(e)
                    item.completed_at = datetime.now()
                    
                    # Schedule retry if under max attempts
                    if item.retry_count < settings.processing.max_retries:
                        delay = self._calculate_retry_delay(item.retry_count)
                        item.next_retry_at = datetime.now() + timedelta(seconds=delay)
                        logger.info(
                            f"Scheduling retry for {item.document_id} "
                            f"in {delay} seconds"
                        )
                    
                    self._completed[item.document_id] = item
                    self._save_item(item)
    
    def _process_queue(self) -> None:
        """Process documents in queue."""
        while not self._stop_event.is_set():
            try:
                # Check resources
                can_process, reason, _ = self._resource_monitor.check_resources()
                if not can_process:
                    logger.warning(f"Pausing queue processing: {reason}")
                    self._stop_event.wait(60)  # Wait 1 minute before checking again
                    continue
                
                # Get next document
                try:
                    item = self._queue.get(timeout=1)
                except Empty:
                    continue
                
                # Move to processing
                with self._lock:
                    if item.document_id in self._items:
                        item = self._items.pop(item.document_id)
                        item.status = ProcessingStatus.PROCESSING
                        item.started_at = datetime.now()
                        self._processing[item.document_id] = item
                        self._save_item(item)
                
                # Process document
                asyncio.run(self._process_document(item))
                
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                self._stop_event.wait(5)  # Wait 5 seconds before retrying