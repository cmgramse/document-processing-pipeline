"""
Document Processing Statistics Module

This module handles statistics tracking and reporting for document processing
operations. It provides functionality to track various metrics about document
processing and generate reports.

The module manages:
- Processing time tracking
- Success/failure rates
- Document and chunk counts
- Resource usage statistics

Features:
- Real-time statistics updates
- Customizable metrics
- Batch processing stats
- Performance monitoring

Example:
    Track processing stats:
        stats = ProcessingStats()
        stats.update(files_processed=1)
        print(stats.to_dict())
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

@dataclass
class ProcessingStats:
    """
    Statistics for document processing operations.
    
    Attributes:
        files_processed: Number of files processed
        chunks_created: Number of chunks created
        embeddings_generated: Number of embeddings generated
        errors: Number of errors encountered
        start_time: Processing start time
        end_time: Processing end time
        batch_sizes: List of batch sizes processed
    
    Example:
        stats = ProcessingStats()
        stats.start()
        # Process documents
        stats.end()
        print(f"Processed {stats.files_processed} files")
    """
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    batch_sizes: list = field(default_factory=list)
    
    def start(self) -> None:
        """
        Start tracking processing time.
        
        Example:
            stats = ProcessingStats()
            stats.start()
        """
        self.start_time = datetime.now()
    
    def end(self) -> None:
        """
        End tracking processing time.
        
        Example:
            stats.end()
            duration = stats.get_duration()
        """
        self.end_time = datetime.now()
    
    def update(self, **kwargs) -> None:
        """
        Update statistics with new values.
        
        Args:
            **kwargs: Keyword arguments with stat updates
        
        Example:
            stats.update(files_processed=5, chunks_created=20)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, (int, float)):
                    setattr(self, key, current + value)
                elif isinstance(current, list):
                    current.append(value)
    
    def get_duration(self) -> Optional[timedelta]:
        """
        Get total processing duration.
        
        Returns:
            timedelta: Processing duration if completed
            None: If processing not completed
        
        Example:
            duration = stats.get_duration()
            print(f"Processing took {duration.total_seconds()} seconds")
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_rate(self) -> Optional[float]:
        """
        Get processing rate in files per second.
        
        Returns:
            float: Files processed per second
            None: If processing not completed
        
        Example:
            rate = stats.get_rate()
            print(f"Processing rate: {rate:.2f} files/second")
        """
        duration = self.get_duration()
        if duration and duration.total_seconds() > 0:
            return self.files_processed / duration.total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary format.
        
        Returns:
            Dict containing all statistics
        
        Example:
            stats_dict = stats.to_dict()
            print(f"Success rate: {stats_dict['success_rate']}%")
        """
        stats = {
            'files_processed': self.files_processed,
            'chunks_created': self.chunks_created,
            'embeddings_generated': self.embeddings_generated,
            'errors': self.errors,
            'success_rate': (
                (self.files_processed - self.errors) / self.files_processed * 100
                if self.files_processed > 0 else 0
            )
        }
        
        if duration := self.get_duration():
            stats['duration_seconds'] = duration.total_seconds()
            stats['processing_rate'] = self.get_rate()
        
        if self.batch_sizes:
            stats['avg_batch_size'] = sum(self.batch_sizes) / len(self.batch_sizes)
        
        return stats
    
    def __str__(self):
        return (
            f"Files processed: {self.files_processed}\n"
            f"Chunks created: {self.chunks_created}\n"
            f"Embeddings generated: {self.embeddings_generated}\n"
            f"Errors: {self.errors}"
        )