"""
Document processing with enhanced settings support and unified queue management.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..config.settings import settings
from ..api.jina import jina_client
from ..database.qdrant import QdrantManager
from ..models.document import Document, ProcessingStats, ProcessingStatus
from .resource_monitor import ResourceMonitor
from .queue_manager import QueueManager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents with unified queue management."""
    
    def __init__(self):
        """Initialize document processor."""
        self.qdrant = QdrantManager()
        self.stats = ProcessingStats(document_id="batch")
        self.resource_monitor = ResourceMonitor()
        self.queue_manager = QueueManager()
    
    def start(self) -> None:
        """Start document processing."""
        self.stats.start()
        self.queue_manager.start()
    
    def stop(self) -> None:
        """Stop document processing."""
        self.stats.end()
        self.queue_manager.stop()
    
    def process_documents(self, documents: List[Document]) -> ProcessingStats:
        """
        Process multiple documents using queue management.
        
        Args:
            documents: List of documents to process
            
        Returns:
            ProcessingStats: Processing statistics
        """
        # Check resources before starting
        can_process, reason, stats = self.resource_monitor.check_resources()
        if not can_process:
            logger.warning(f"Cannot start processing: {reason}")
            self.stats.skipped_documents += len(documents)
            return self.stats
        
        # Update resource stats
        self.stats.system_memory_percent = stats['system_memory_percent']
        self.stats.process_memory_mb = stats['process_memory_mb']
        
        # Add documents to queue
        self.queue_manager.add_documents(documents)
        self.stats.total_documents += len(documents)
        
        return self.stats
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status.
        
        Returns:
            Dict containing:
            - pending: Number of pending documents
            - processing: Number of documents being processed
            - completed: Number of completed documents
            - failed: Number of failed documents
            - recent_updates: List of recent status updates
        """
        statuses = self.queue_manager.get_all_statuses()
        
        # Count documents by status
        counts = {
            'pending': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        recent_updates = []
        
        for doc_id, status in statuses.items():
            counts[status['status']] += 1
            
            # Add to recent updates if status changed in last minute
            if status.get('completed_at') or status.get('started_at'):
                recent_updates.append({
                    'filename': doc_id,
                    'status': status['status'],
                    'error': status['error']
                })
        
        # Sort recent updates by timestamp
        recent_updates.sort(
            key=lambda x: x.get('completed_at') or x.get('started_at'),
            reverse=True
        )
        
        return {
            'pending': counts['pending'],
            'processing': counts['processing'],
            'completed': counts['completed'],
            'failed': counts['failed'],
            'skipped': counts['skipped'],
            'recent_updates': recent_updates[:5]  # Last 5 updates
        }
    
    def verify_processing(self, timeout_minutes: int = 5) -> bool:
        """
        Verify all documents were processed successfully.
        
        Args:
            timeout_minutes: Maximum time to wait for processing to complete
            
        Returns:
            bool: True if all documents processed successfully
        """
        import time
        start_time = time.time()
        timeout = timeout_minutes * 60
        
        while time.time() - start_time < timeout:
            status = self.get_processing_status()
            
            # Check if all documents are processed
            if status['pending'] == 0 and status['processing'] == 0:
                return status['failed'] == 0 and status['skipped'] == 0
            
            time.sleep(5)  # Wait 5 seconds before checking again
        
        return False  # Timeout reached
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing stats."""
        return self.stats 