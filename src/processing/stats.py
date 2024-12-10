"""
Document Processing Statistics Module

This module provides centralized statistics tracking for the document processing pipeline.
It tracks various metrics across different stages of processing including document chunking,
embedding generation, and vector storage operations.

Key Features:
- Processing time tracking for each stage
- Success/failure rates by operation type
- Document and chunk counts with status tracking
- Resource usage statistics
- API call metrics
- Database operation tracking
- Vector store synchronization stats

Usage Examples:

1. Basic Usage:
    ```python
    stats = ProcessingStats()
    stats.start()
    # Process documents
    stats.update(files_processed=1, chunks_created=5)
    stats.end()
    print(stats.to_dict())
    ```

2. Track API Calls:
    ```python
    stats.track_api_call('jina_embedding', success=True, latency=0.5)
    stats.track_api_call('qdrant_upload', success=False, error="Connection failed")
    ```

3. Track Database Operations:
    ```python
    stats.track_db_operation('chunk_insert', count=10)
    stats.track_db_operation('status_update', success=True)
    ```
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

@dataclass
class APIStats:
    """Statistics for API operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)
    
    def track_call(self, success: bool, latency: float = 0.0, error: str = None) -> None:
        """Track an API call result."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.errors[error] = self.errors.get(error, 0) + 1
        self.total_latency += latency
    
    def get_avg_latency(self) -> float:
        """Get average API call latency."""
        return self.total_latency / self.total_calls if self.total_calls > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': (self.successful_calls / self.total_calls * 100 
                           if self.total_calls > 0 else 0),
            'avg_latency': self.get_avg_latency(),
            'errors': self.errors
        }

@dataclass
class DBStats:
    """Statistics for database operations."""
    operations: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)
    
    def track_operation(self, operation: str, count: int = 1, error: str = None) -> None:
        """Track a database operation."""
        self.operations[operation] = self.operations.get(operation, 0) + count
        if error:
            self.errors[error] = self.errors.get(error, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'operations': self.operations,
            'total_operations': sum(self.operations.values()),
            'errors': self.errors,
            'error_rate': (len(self.errors) / sum(self.operations.values()) * 100 
                         if sum(self.operations.values()) > 0 else 0)
        }

@dataclass
class ProcessingStats:
    """
    Comprehensive statistics for document processing operations.
    
    Attributes:
        files_processed: Number of files processed
        chunks_created: Number of chunks created
        embeddings_generated: Number of embeddings generated
        vectors_stored: Number of vectors stored in Qdrant
        errors: Number of errors encountered
        start_time: Processing start time
        end_time: Processing end time
        api_stats: Detailed API call statistics
        db_stats: Database operation statistics
        batch_sizes: List of batch sizes processed
    """
    # Document processing stats
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    errors: int = 0
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Detailed stats
    api_stats: Dict[str, APIStats] = field(default_factory=lambda: {
        'jina_embedding': APIStats(),
        'jina_chunking': APIStats(),
        'qdrant_upload': APIStats()
    })
    db_stats: DBStats = field(default_factory=DBStats)
    batch_sizes: List[int] = field(default_factory=list)
    
    def start(self) -> None:
        """Start tracking processing time."""
        self.start_time = datetime.now()
    
    def end(self) -> None:
        """End tracking processing time."""
        self.end_time = datetime.now()
    
    def update(self, **kwargs) -> None:
        """
        Update statistics with new values.
        
        Args:
            **kwargs: Stat updates (e.g., files_processed=1, chunks_created=5)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, (int, float)):
                    setattr(self, key, current + value)
                elif isinstance(current, list):
                    current.append(value)
    
    def track_api_call(self, api: str, success: bool, latency: float = 0.0, 
                      error: str = None) -> None:
        """
        Track an API call result.
        
        Args:
            api: API name ('jina_embedding', 'jina_chunking', 'qdrant_upload')
            success: Whether the call succeeded
            latency: API call latency in seconds
            error: Error message if failed
        """
        if api in self.api_stats:
            self.api_stats[api].track_call(success, latency, error)
    
    def track_db_operation(self, operation: str, count: int = 1, 
                         error: str = None) -> None:
        """
        Track a database operation.
        
        Args:
            operation: Operation name (e.g., 'chunk_insert', 'status_update')
            count: Number of operations performed
            error: Error message if failed
        """
        self.db_stats.track_operation(operation, count, error)
    
    def get_duration(self) -> Optional[timedelta]:
        """Get total processing duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_rate(self) -> Optional[float]:
        """Get processing rate in files per second."""
        duration = self.get_duration()
        if duration and duration.total_seconds() > 0:
            return self.files_processed / duration.total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all statistics to dictionary format.
        
        Returns:
            Dict containing all statistics including:
            - Document processing stats
            - API call stats
            - Database operation stats
            - Timing and rate information
        """
        stats = {
            'document_processing': {
                'files_processed': self.files_processed,
                'chunks_created': self.chunks_created,
                'embeddings_generated': self.embeddings_generated,
                'vectors_stored': self.vectors_stored,
                'errors': self.errors,
                'success_rate': (
                    (self.files_processed - self.errors) / self.files_processed * 100
                    if self.files_processed > 0 else 0
                )
            },
            'api_calls': {
                name: api_stats.to_dict()
                for name, api_stats in self.api_stats.items()
            },
            'database': self.db_stats.to_dict()
        }
        
        if duration := self.get_duration():
            stats['timing'] = {
                'duration_seconds': duration.total_seconds(),
                'processing_rate': self.get_rate(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None
            }
        
        if self.batch_sizes:
            stats['batching'] = {
                'total_batches': len(self.batch_sizes),
                'avg_batch_size': sum(self.batch_sizes) / len(self.batch_sizes),
                'min_batch_size': min(self.batch_sizes),
                'max_batch_size': max(self.batch_sizes)
            }
        
        return stats
    
    def to_json(self) -> str:
        """Convert statistics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [
            f"Document Processing Stats:",
            f"- Files processed: {self.files_processed}",
            f"- Chunks created: {self.chunks_created}",
            f"- Embeddings generated: {self.embeddings_generated}",
            f"- Vectors stored: {self.vectors_stored}",
            f"- Errors: {self.errors}"
        ]
        
        if duration := self.get_duration():
            parts.extend([
                f"\nTiming:",
                f"- Duration: {duration.total_seconds():.2f}s",
                f"- Rate: {self.get_rate():.2f} files/second"
            ])
        
        return "\n".join(parts)