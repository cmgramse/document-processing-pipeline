"""
Database Maintenance Module

This module handles database maintenance operations for the document management
system. It provides functionality for cleaning up old data, optimizing storage,
and ensuring database health.

The module manages:
- Cleanup of old chunks and documents
- Version tracking and upgrades
- Storage optimization
- Data consistency checks

Features:
- Configurable retention periods
- Safe deletion with transaction support
- Version-based cleanup
- Storage space reclamation

Example:
    Clean up old data:
        stats = cleanup_database(conn, retention_days=30)
    
    Optimize storage:
        optimize_storage(conn)
    
    Track chunk versions:
        track_chunk_versions(conn)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sqlite3

def cleanup_database(conn: sqlite3.Connection, retention_days: int = 30) -> Dict[str, int]:
    """Clean up old and failed chunks from the database."""
    stats = {
        'old_chunks_removed': 0,
        'failed_chunks_removed': 0
    }
    
    c = conn.cursor()
    try:
        # Delete old chunks that are not referenced in documents
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        # First, identify chunks that can be safely deleted
        c.execute("""
            SELECT c.id 
            FROM chunks c
            LEFT JOIN documents d ON c.id = d.chunk_id
            WHERE c.created_at < ?
            AND (d.chunk_id IS NULL OR d.qdrant_status = 'uploaded')
        """, (cutoff_date,))
        
        old_chunks = [row[0] for row in c.fetchall()]
        
        if old_chunks:
            placeholders = ','.join('?' * len(old_chunks))
            c.execute(f"""
                DELETE FROM chunks 
                WHERE id IN ({placeholders})
            """, old_chunks)
            stats['old_chunks_removed'] = c.rowcount

        # Delete failed chunks that are not referenced
        c.execute("""
            DELETE FROM chunks 
            WHERE embedding_status = 'failed'
            AND id NOT IN (
                SELECT chunk_id 
                FROM documents 
                WHERE chunk_id IS NOT NULL
                AND qdrant_status != 'uploaded'
            )
        """)
        stats['failed_chunks_removed'] = c.rowcount

        conn.commit()
        
        # Run VACUUM in a separate transaction
        conn.isolation_level = None
        c.execute('VACUUM')
        conn.isolation_level = ''  # Reset to default
        
        return stats
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during database cleanup: {str(e)}")
        raise

def optimize_batch_processing(data, batch_size: int = 50, max_tokens_per_batch: int = 8000):
    """
    Optimize processing by batching items.
    
    Args:
        data: Either a SQLite connection or a list of items to batch
        batch_size: Size of processing batches
        max_tokens_per_batch: Maximum number of tokens per batch (approx 4 chars per token)
        
    Returns:
        If data is a list: Generator yielding batches of items
        If data is a connection: None (optimizes database settings)
    """
    if isinstance(data, list):
        # Handle list input
        current_batch = []
        current_token_count = 0
        
        for item in data:
            # Estimate token count (rough approximation: 4 chars per token)
            item_token_count = len(str(item)) // 4
            
            # If adding this item would exceed either limit, yield current batch
            if (len(current_batch) >= batch_size or 
                current_token_count + item_token_count > max_tokens_per_batch):
                if current_batch:  # Don't yield empty batches
                    yield current_batch
                current_batch = []
                current_token_count = 0
            
            current_batch.append(item)
            current_token_count += item_token_count
        
        # Yield any remaining items
        if current_batch:
            yield current_batch
    else:
        # Handle database connection
        c = data.cursor()
        
        try:
            # Set journal mode first as it requires a separate transaction
            data.isolation_level = None  # Required for some PRAGMA changes
            c.execute('PRAGMA journal_mode = MEMORY')
            
            # Set and verify other pragmas
            pragmas = {
                'synchronous': 2,  # NORMAL - minimum safe level
                'temp_store': 2,  # MEMORY
                'cache_size': 10000
            }
            
            for name, value in pragmas.items():
                c.execute(f'PRAGMA {name} = {value}')
                c.execute(f'PRAGMA {name}')
                actual = c.fetchone()[0]
                if actual != value:
                    logging.warning(f"Failed to set {name} to {value}, got {actual}")
            
            # Create temporary indexes if needed
            c.execute('''CREATE INDEX IF NOT EXISTS 
                        idx_chunks_status ON chunks(embedding_status)''')
            c.execute('''CREATE INDEX IF NOT EXISTS 
                        idx_docs_status ON documents(qdrant_status)''')
            
            data.isolation_level = ''  # Reset to default
            data.commit()
            logging.info("Batch processing optimizations applied")
        except Exception as e:
            if data.isolation_level is None:
                data.isolation_level = ''  # Reset if exception occurs
            data.rollback()
            logging.error(f"Failed to apply optimizations: {str(e)}")
            raise

def track_chunk_versions(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Track and manage chunk versions in the database.
    
    Args:
        conn: SQLite database connection
    
    Returns:
        Dict containing version statistics:
        - current_version: Latest schema version
        - chunks_updated: Number of chunks updated
        - errors: Number of errors encountered
    
    Example:
        stats = track_chunk_versions(conn)
        print(f"Updated {stats['chunks_updated']} chunks")
    """
    c = conn.cursor()
    stats = {'current_version': 0, 'chunks_updated': 0, 'errors': 0}
    
    try:
        # Get current version
        c.execute('SELECT MAX(version) FROM chunks')
        current_version = c.fetchone()[0] or 0
        stats['current_version'] = current_version
        
        # Update unversioned chunks
        c.execute('''
        UPDATE chunks 
        SET version = ? 
        WHERE version IS NULL OR version = 0
        ''', (current_version,))
        stats['chunks_updated'] = c.rowcount
        
        conn.commit()
        logging.info(f"Version tracking completed: {stats}")
        
    except Exception as e:
        conn.rollback()
        stats['errors'] += 1
        logging.error(f"Version tracking failed: {str(e)}")
        raise
    
    return stats
