"""
Document Manager Module

Handles document operations using the new schema:
- Document creation and updates
- Status tracking
- Processing queue management
- History tracking
"""

import os
import uuid
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sqlite3

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document operations with the new schema."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        
    def _get_db(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def create_document(self, filename: str, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new document record.
        
        Args:
            filename: Document filename
            content: Document content
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        doc_id = hashlib.md5(filename.encode()).hexdigest()
        
        with self._get_db() as conn:
            # Create document
            conn.execute("""
                INSERT INTO documents (
                    id, filename, content, status, metadata
                ) VALUES (?, ?, ?, 'pending', ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    status = 'pending',
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, (doc_id, filename, content, json.dumps(metadata or {})))
            
            # Create initial processing task
            task_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO processing_queue (
                    id, document_id, task_type, status
                ) VALUES (?, ?, 'chunk', 'pending')
            """, (task_id, doc_id))
            
        return doc_id
        
    def create_chunk(self, doc_id: str, chunk_id: str, content: str,
                    chunk_index: int, token_count: int, qdrant_id: str) -> bool:
        """
        Create a new chunk record.
        
        Args:
            doc_id: Parent document ID
            chunk_id: Chunk ID (UUID)
            content: Chunk content
            chunk_index: Position in document
            token_count: Number of tokens
            qdrant_id: Pre-generated Qdrant ID
            
        Returns:
            True if successful
        """
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO chunks (
                    id, document_id, content, chunk_index,
                    token_count, qdrant_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (chunk_id, doc_id, content, chunk_index, token_count, qdrant_id))
            return True
            
    def update_chunk(self, chunk_id: str, status: Optional[str] = None,
                    embedding: Optional[List[float]] = None,
                    error: Optional[str] = None) -> bool:
        """
        Update chunk record.
        
        Args:
            chunk_id: Chunk ID
            status: New status
            embedding: Vector embedding
            error: Error message
            
        Returns:
            True if successful
        """
        with self._get_db() as conn:
            updates = []
            params = []
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                
            if embedding is not None:
                updates.append("embedding = ?")
                params.append(json.dumps(embedding))
                
            if error is not None:
                updates.append("error_message = ?")
                params.append(error)
                
            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(chunk_id)
                
                conn.execute(f"""
                    UPDATE chunks
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                return True
            return False
            
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with self._get_db() as conn:
            row = conn.execute("""
                SELECT 
                    d.*,
                    COUNT(DISTINCT c.id) as chunk_count,
                    COUNT(DISTINCT CASE WHEN c.status = 'completed' THEN c.id END) as completed_chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = ?
                GROUP BY d.id
            """, (doc_id,)).fetchone()
            
            if row:
                doc = dict(row)
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                return doc
            return None
            
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document by filename."""
        with self._get_db() as conn:
            row = conn.execute("""
                SELECT 
                    d.*,
                    COUNT(DISTINCT c.id) as chunk_count,
                    COUNT(DISTINCT CASE WHEN c.status = 'completed' THEN c.id END) as completed_chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.filename = ?
                GROUP BY d.id
            """, (filename,)).fetchone()
            
            if row:
                doc = dict(row)
                doc['metadata'] = json.loads(doc['metadata']) if doc['metadata'] else {}
                return doc
            return None
            
    def update_document_status(self, doc_id: str, status: str, error: Optional[str] = None) -> bool:
        """Update document status."""
        with self._get_db() as conn:
            conn.execute("""
                UPDATE documents 
                SET status = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, error, doc_id))
            return True
            
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with self._get_db() as conn:
            rows = conn.execute("""
                SELECT *
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (doc_id,)).fetchall()
            
            chunks = []
            for row in rows:
                chunk = dict(row)
                chunk['embedding'] = json.loads(chunk['embedding']) if chunk['embedding'] else None
                chunks.append(chunk)
            return chunks
            
    def get_pending_tasks(self, task_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending processing tasks."""
        with self._get_db() as conn:
            query = """
                SELECT q.*, d.filename
                FROM processing_queue q
                JOIN documents d ON d.id = q.document_id
                WHERE q.status = 'pending'
            """
            
            if task_type:
                query += " AND q.task_type = ?"
                rows = conn.execute(query + " ORDER BY q.priority DESC, q.created_at LIMIT ?",
                    (task_type, limit)).fetchall()
            else:
                rows = conn.execute(query + " ORDER BY q.priority DESC, q.created_at LIMIT ?",
                    (limit,)).fetchall()
                
            return [dict(row) for row in rows]
            
    def update_task_status(self, task_id: str, status: str, error: Optional[str] = None) -> bool:
        """Update task status."""
        with self._get_db() as conn:
            conn.execute("""
                UPDATE processing_queue
                SET status = ?,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, error, task_id))
            return True
            
    def add_processing_history(self, doc_id: str, action: str, 
                             chunk_id: Optional[str] = None,
                             details: Optional[Dict] = None) -> bool:
        """Add processing history entry."""
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO processing_history (
                    document_id, chunk_id, action, status, details
                ) VALUES (?, ?, ?, 'success', ?)
            """, (doc_id, chunk_id, action, json.dumps(details or {})))
            return True
            
    def get_processing_history(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get processing history for a document."""
        with self._get_db() as conn:
            rows = conn.execute("""
                SELECT *
                FROM processing_history
                WHERE document_id = ?
                ORDER BY created_at DESC
            """, (doc_id,)).fetchall()
            
            history = []
            for row in rows:
                entry = dict(row)
                entry['details'] = json.loads(entry['details']) if entry['details'] else {}
                history.append(entry)
            return history
            
    def cleanup_failed_tasks(self, max_retries: int = 3) -> int:
        """Clean up failed tasks and reset for retry."""
        with self._get_db() as conn:
            # Get failed tasks under retry limit
            rows = conn.execute("""
                UPDATE processing_queue
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'failed'
                AND retry_count < ?
                RETURNING id
            """, (max_retries,)).fetchall()
            
            # Mark permanently failed tasks
            conn.execute("""
                UPDATE processing_queue
                SET status = 'permanent_failure',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'failed'
                AND retry_count >= ?
            """, (max_retries,))
            
            return len(rows)
            
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._get_db() as conn:
            stats = {}
            
            # Document stats
            doc_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed
                FROM documents
            """).fetchone()
            stats['documents'] = dict(doc_stats)
            
            # Chunk stats
            chunk_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as failed
                FROM chunks
            """).fetchone()
            stats['chunks'] = dict(chunk_stats)
            
            # Queue stats
            queue_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                    COUNT(CASE WHEN status = 'permanent_failure' THEN 1 END) as permanent_failures
                FROM processing_queue
            """).fetchone()
            stats['queue'] = dict(queue_stats)
            
            return stats
