"""
Document Manager Module

Handles database operations for document storage and retrieval.
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from .connection import get_connection_pool, transaction, get_db_connection
from ..models.document import Document, ProcessingStats, ProcessingStatus
from ..vector_store import get_vector_store

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document storage and retrieval in the SQLite database."""
    
    def __init__(self):
        """Initialize the document manager."""
        self._pool = get_connection_pool()
        self._vector_store = get_vector_store()
    
    def create_document(self, document: Document) -> None:
        """Create a new document in the database."""
        with get_db_connection(self._pool) as conn:
            with transaction(conn):
                conn.connection.execute(
                    """
                    INSERT INTO documents (
                        id, title, content, metadata, vector_id,
                        created_at, updated_at, processing_status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.title,
                        document.content,
                        json.dumps(document.metadata),
                        document.vector_id,
                        document.created_at.isoformat() if document.created_at else None,
                        document.updated_at.isoformat() if document.updated_at else None,
                        document.processing_status.value,
                        document.error_message
                    )
                )
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        with get_db_connection(self._pool) as conn:
            cursor = conn.connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            if row:
                return Document(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    vector_id=row['vector_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    processing_status=ProcessingStatus(row['processing_status']),
                    error_message=row['error_message']
                )
        return None
    
    def update_document(self, document: Document) -> None:
        """Update an existing document."""
        with get_db_connection(self._pool) as conn:
            with transaction(conn):
                conn.connection.execute(
                    """
                    UPDATE documents SET
                        title = ?,
                        content = ?,
                        metadata = ?,
                        vector_id = ?,
                        updated_at = ?,
                        processing_status = ?,
                        error_message = ?
                    WHERE id = ?
                    """,
                    (
                        document.title,
                        document.content,
                        json.dumps(document.metadata),
                        document.vector_id,
                        datetime.now().isoformat(),
                        document.processing_status.value,
                        document.error_message,
                        document.id
                    )
                )
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        with get_db_connection(self._pool) as conn:
            with transaction(conn):
                cursor = conn.connection.execute(
                    "DELETE FROM documents WHERE id = ?",
                    (document_id,)
                )
                return cursor.rowcount > 0
    
    def get_all_documents(self) -> List[Document]:
        """Retrieve all documents."""
        with get_db_connection(self._pool) as conn:
            cursor = conn.connection.execute(
                "SELECT * FROM documents ORDER BY created_at"
            )
            return [
                Document(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    vector_id=row['vector_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    processing_status=ProcessingStatus(row['processing_status']),
                    error_message=row['error_message']
                )
                for row in cursor.fetchall()
            ]
    
    def get_pending_documents(self) -> List[Document]:
        """Retrieve all documents with pending processing status."""
        with get_db_connection(self._pool) as conn:
            cursor = conn.connection.execute(
                "SELECT * FROM documents WHERE processing_status = ? ORDER BY created_at",
                (ProcessingStatus.PENDING.value,)
            )
            return [
                Document(
                    id=row['id'],
                    title=row['title'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    vector_id=row['vector_id'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    processing_status=ProcessingStatus(row['processing_status']),
                    error_message=row['error_message']
                )
                for row in cursor.fetchall()
            ]
    
    def record_processing_stats(self, stats: ProcessingStats) -> None:
        """Record document processing statistics."""
        with get_db_connection(self._pool) as conn:
            with transaction(conn):
                conn.connection.execute(
                    """
                    INSERT INTO processing_stats (
                        document_id, processing_time, tokens_processed, chunks_created,
                        total_documents, processed_documents, failed_documents,
                        skipped_documents, total_chunks, completed_chunks,
                        failed_chunks, total_embeddings, failed_embeddings,
                        system_memory_percent, process_memory_mb, batch_sizes,
                        errors, retries, start_time, end_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stats.document_id,
                        stats.processing_time,
                        stats.tokens_processed,
                        stats.chunks_created,
                        stats.total_documents,
                        stats.processed_documents,
                        stats.failed_documents,
                        stats.skipped_documents,
                        stats.total_chunks,
                        stats.completed_chunks,
                        stats.failed_chunks,
                        stats.total_embeddings,
                        stats.failed_embeddings,
                        stats.system_memory_percent,
                        stats.process_memory_mb,
                        json.dumps(stats.batch_sizes),
                        json.dumps(stats.errors),
                        stats.retries,
                        stats.start_time.isoformat() if stats.start_time else None,
                        stats.end_time.isoformat() if stats.end_time else None
                    )
                )
    
    def get_processing_stats(self, document_id: str) -> List[ProcessingStats]:
        """Get processing statistics for a document."""
        with get_db_connection(self._pool) as conn:
            cursor = conn.connection.execute(
                "SELECT * FROM processing_stats WHERE document_id = ? ORDER BY timestamp",
                (document_id,)
            )
            return [
                ProcessingStats(
                    document_id=row['document_id'],
                    processing_time=row['processing_time'],
                    tokens_processed=row['tokens_processed'],
                    chunks_created=row['chunks_created'],
                    total_documents=row['total_documents'],
                    processed_documents=row['processed_documents'],
                    failed_documents=row['failed_documents'],
                    skipped_documents=row['skipped_documents'],
                    total_chunks=row['total_chunks'],
                    completed_chunks=row['completed_chunks'],
                    failed_chunks=row['failed_chunks'],
                    total_embeddings=row['total_embeddings'],
                    failed_embeddings=row['failed_embeddings'],
                    system_memory_percent=row['system_memory_percent'],
                    process_memory_mb=row['process_memory_mb'],
                    batch_sizes=json.loads(row['batch_sizes']) if row['batch_sizes'] else [],
                    errors=json.loads(row['errors']) if row['errors'] else [],
                    retries=row['retries'],
                    start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None
                )
                for row in cursor.fetchall()
            ] 
    
    def sync_with_vector_store(self, document_id: str) -> None:
        """
        Synchronize document status with vector store.
        Updates the document status based on whether its vector exists in the store.
        """
        document = self.get_document(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found for vector store sync")
            return

        # Check if vector exists in store by attempting to retrieve similar documents
        vector_exists = False
        if document.vector_id:
            try:
                similar_docs = self._vector_store.similarity_search_with_score(
                    query="",  # Empty query just to check existence
                    k=1,
                    filter={"id": document.vector_id}
                )
                vector_exists = len(similar_docs) > 0
            except Exception as e:
                logger.error(f"Error checking vector existence: {str(e)}")
                return
        
        if vector_exists and document.processing_status != ProcessingStatus.COMPLETED:
            document.processing_status = ProcessingStatus.COMPLETED
            document.error_message = None
            self.update_document(document)
            logger.info(f"Updated document {document_id} status to COMPLETED based on vector store sync")
        elif not vector_exists and document.processing_status == ProcessingStatus.COMPLETED:
            document.processing_status = ProcessingStatus.PENDING
            document.error_message = "Vector not found in store"
            self.update_document(document)
            logger.warning(f"Reset document {document_id} to PENDING due to missing vector")

    def check_vector_store_consistency(self) -> Dict[str, Any]:
        """
        Check consistency between SQLite and vector store.
        Returns a dictionary with consistency statistics.
        """
        documents = self.get_all_documents()
        stats = {
            "total_documents": len(documents),
            "completed_count": 0,
            "missing_vectors": 0,
            "inconsistent_status": 0,
            "documents_needing_sync": []
        }

        for doc in documents:
            if doc.processing_status == ProcessingStatus.COMPLETED:
                stats["completed_count"] += 1
                vector_exists = False
                if doc.vector_id:
                    try:
                        similar_docs = self._vector_store.similarity_search_with_score(
                            query="",
                            k=1,
                            filter={"id": doc.vector_id}
                        )
                        vector_exists = len(similar_docs) > 0
                    except Exception as e:
                        logger.error(f"Error checking vector existence for {doc.id}: {str(e)}")
                        continue

                if not doc.vector_id or not vector_exists:
                    stats["missing_vectors"] += 1
                    stats["documents_needing_sync"].append(doc.id)
                    stats["inconsistent_status"] += 1

        return stats

    def update_vector_store_status(self, document_id: str, success: bool, error_message: Optional[str] = None) -> None:
        """
        Update document status after vector store operation.
        Args:
            document_id: The ID of the document to update
            success: Whether the vector store operation was successful
            error_message: Optional error message if the operation failed
        """
        document = self.get_document(document_id)
        if not document:
            logger.error(f"Cannot update vector store status: Document {document_id} not found")
            return

        if success:
            document.processing_status = ProcessingStatus.COMPLETED
            document.error_message = None
        else:
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = error_message or "Vector store operation failed"

        self.update_document(document)
        logger.info(f"Updated document {document_id} status to {document.processing_status.value}") 