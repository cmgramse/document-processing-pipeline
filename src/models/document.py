"""
Document and Processing Statistics Models

This module defines the core data models used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class ProcessingStatus(Enum):
    """
    Processing status enum used across all processing operations.
    This is the single source of truth for status values.
    """
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'  # Used only in batch processing
    
    @classmethod
    def get_database_check_constraint(cls) -> str:
        """Get SQL CHECK constraint for status fields."""
        values = [f"'{status.value}'" for status in cls]
        return f"CHECK (processing_status IN ({', '.join(values)}))"
    
    @classmethod
    def get_batch_check_constraint(cls) -> str:
        """Get SQL CHECK constraint for batch status fields."""
        values = [f"'{status.value}'" for status in cls]
        return f"CHECK (status IN ({', '.join(values)}))"

@dataclass
class Document:
    """
    Represents a document in the system.
    
    This is the central model that represents a document throughout its lifecycle,
    from initial creation through processing and vector storage.
    """
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Ensure proper types after initialization."""
        if isinstance(self.processing_status, str):
            self.processing_status = ProcessingStatus(self.processing_status)
        
        if isinstance(self.metadata, str):
            import json
            self.metadata = json.loads(self.metadata)
        
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at.replace('Z', '+00:00'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'vector_id': self.vector_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processing_status': self.processing_status.value,
            'error_message': self.error_message
        }

@dataclass
class ProcessingStats:
    """
    Statistics for document processing operations.
    Single source of truth for processing statistics.
    
    Attributes:
        document_id: ID of the document being processed
        processing_time: Time taken for processing in seconds
        tokens_processed: Number of tokens processed
        chunks_created: Number of chunks created
        total_documents: Total number of documents in batch/session
        processed_documents: Number of successfully processed documents
        failed_documents: Number of failed documents
        skipped_documents: Number of skipped documents
        total_chunks: Total number of chunks created
        completed_chunks: Number of successfully processed chunks
        failed_chunks: Number of failed chunks
        total_embeddings: Total number of embeddings generated
        failed_embeddings: Number of failed embedding generations
        system_memory_percent: System memory usage percentage
        process_memory_mb: Process memory usage in MB
        batch_sizes: List of batch sizes used
        errors: List of error messages
        retries: Number of retry attempts
        start_time: Processing start time
        end_time: Processing end time
    """
    document_id: str
    processing_time: float = 0.0
    tokens_processed: int = 0
    chunks_created: int = 0
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    total_embeddings: int = 0
    failed_embeddings: int = 0
    system_memory_percent: float = 0.0
    process_memory_mb: float = 0.0
    batch_sizes: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    retries: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start tracking processing time."""
        self.start_time = datetime.now()
    
    def end(self) -> None:
        """End tracking processing time."""
        self.end_time = datetime.now()
        if self.start_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
    
    def add_error(self, error: str) -> None:
        """Add an error message to the stats."""
        self.errors.append(error)
    
    def add_batch_size(self, size: int) -> None:
        """Add a batch size to the stats."""
        self.batch_sizes.append(size)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    @property
    def average_batch_size(self) -> float:
        """Calculate average batch size."""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            'document_id': self.document_id,
            'processing_time': self.processing_time,
            'tokens_processed': self.tokens_processed,
            'chunks_created': self.chunks_created,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'failed_documents': self.failed_documents,
            'skipped_documents': self.skipped_documents,
            'total_chunks': self.total_chunks,
            'completed_chunks': self.completed_chunks,
            'failed_chunks': self.failed_chunks,
            'total_embeddings': self.total_embeddings,
            'failed_embeddings': self.failed_embeddings,
            'system_memory_percent': self.system_memory_percent,
            'process_memory_mb': self.process_memory_mb,
            'batch_sizes': self.batch_sizes,
            'errors': self.errors,
            'retries': self.retries,
            'success_rate': self.success_rate,
            'average_batch_size': self.average_batch_size,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }
    
    def __str__(self) -> str:
        """Get string representation of stats."""
        return (
            f"Processing Stats for {self.document_id}:\n"
            f"Documents: {self.processed_documents}/{self.total_documents} "
            f"(failed: {self.failed_documents}, skipped: {self.skipped_documents})\n"
            f"Chunks: {self.completed_chunks}/{self.total_chunks} "
            f"(failed: {self.failed_chunks})\n"
            f"Embeddings: {self.total_embeddings - self.failed_embeddings}/"
            f"{self.total_embeddings} (failed: {self.failed_embeddings})\n"
            f"Success Rate: {self.success_rate:.1f}%\n"
            f"Processing Time: {self.processing_time:.2f}s\n"
            f"Memory Usage: {self.process_memory_mb:.1f}MB "
            f"(System: {self.system_memory_percent:.1f}%)"
        )
  