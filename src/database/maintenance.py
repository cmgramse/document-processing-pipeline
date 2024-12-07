"""Database maintenance and optimization utilities."""
import logging
from datetime import datetime
from typing import List, Generator, Dict, Any
import hashlib
from dataclasses import dataclass

@dataclass
class BatchStats:
    total_chunks: int
    total_tokens: int
    batch_count: int
    avg_tokens_per_batch: float

def cleanup_database(conn, retention_days: int = 30) -> Dict[str, int]:
    """
    Clean up old and unnecessary data from the database.
    
    Args:
        conn: SQLite connection
        retention_days: Number of days to retain processed chunks
        
    Returns:
        Dict with cleanup statistics
    """
    logging.info(f"Starting database cleanup with {retention_days} days retention")
    c = conn.cursor()
    stats = {"chunks_removed": 0, "failed_chunks_removed": 0}
    
    try:
        # Remove old successful chunks while keeping their embeddings
        c.execute("""
            DELETE FROM chunks 
            WHERE embedding_status = 'completed'
            AND created_at < datetime('now', ?) 
            AND EXISTS (
                SELECT 1 FROM documents d 
                WHERE d.chunk_id = chunks.id 
                AND d.qdrant_status = 'uploaded'
            )
        """, (f'-{retention_days} days',))
        stats["chunks_removed"] = c.rowcount
        
        # Clean up repeatedly failed chunks
        c.execute("""
            DELETE FROM chunks 
            WHERE embedding_status = 'failed'
            AND (
                SELECT COUNT(*) FROM chunks c2 
                WHERE c2.filename = chunks.filename 
                AND c2.chunk_number = chunks.chunk_number
            ) >= 3
        """)
        stats["failed_chunks_removed"] = c.rowcount
        
        # Vacuum to reclaim space
        c.execute("VACUUM")
        conn.commit()
        
        logging.info(f"Cleanup completed: {stats}")
        return stats
        
    except Exception as e:
        logging.error(f"Error during database cleanup: {str(e)}")
        conn.rollback()
        raise

def track_chunk_versions(conn) -> None:
    """
    Add versioning to chunks to detect content changes.
    
    Args:
        conn: SQLite connection
    """
    logging.info("Setting up chunk version tracking")
    c = conn.cursor()
    
    try:
        # Add content hash column if it doesn't exist
        c.execute("""
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS 
            content_hash TEXT GENERATED ALWAYS AS (
                hex(substr(sha1(content), 1, 8))
            ) STORED
        """)
        
        # Create index for version comparison
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_version 
            ON chunks(filename, chunk_number, content_hash)
        """)
        
        conn.commit()
        logging.info("Successfully set up chunk version tracking")
        
    except Exception as e:
        logging.error(f"Error setting up chunk version tracking: {str(e)}")
        conn.rollback()
        raise

def optimize_batch_processing(chunks: List[str], batch_size: int = 50) -> Generator[List[str], None, BatchStats]:
    """
    Smart batching based on token counts.
    
    Args:
        chunks: List of text chunks to batch
        batch_size: Maximum number of chunks per batch
        
    Yields:
        Batches of chunks
        
    Returns:
        BatchStats with processing statistics
    """
    current_batch = []
    current_tokens = 0
    max_tokens = 8000  # Jina's limit
    
    total_tokens = 0
    total_chunks = len(chunks)
    batch_count = 0
    
    for chunk in chunks:
        chunk_tokens = len(chunk.split())
        total_tokens += chunk_tokens
        
        if current_tokens + chunk_tokens > max_tokens or len(current_batch) >= batch_size:
            batch_count += 1
            yield current_batch
            current_batch = []
            current_tokens = 0
            
        current_batch.append(chunk)
        current_tokens += chunk_tokens
    
    if current_batch:
        batch_count += 1
        yield current_batch
    
    # Return statistics
    return BatchStats(
        total_chunks=total_chunks,
        total_tokens=total_tokens,
        batch_count=batch_count,
        avg_tokens_per_batch=total_tokens / batch_count if batch_count > 0 else 0
    )
