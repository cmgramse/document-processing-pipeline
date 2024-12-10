"""
Database transaction management module.

This module provides utilities for safe and consistent database transactions,
particularly focused on chunk processing operations.
"""

import logging
from contextlib import contextmanager
import sqlite3
from typing import Optional, Any, Generator
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def refresh_schema(conn: sqlite3.Connection) -> None:
    """Force SQLite to reload the schema cache."""
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA schema_version")
        cursor.close()
    except Exception as e:
        logger.error(f"Failed to refresh schema: {e}")
        raise

def init_connection(conn: sqlite3.Connection) -> sqlite3.Cursor:
    """Initialize database connection with proper settings and return cursor."""
    conn.row_factory = sqlite3.Row
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Set journal mode to WAL for better concurrency
    conn.execute("PRAGMA journal_mode = WAL")
    
    # Set synchronous mode to NORMAL for better performance while maintaining safety
    conn.execute("PRAGMA synchronous = NORMAL")
    
    # Ensure schema is fresh
    refresh_schema(conn)
    
    return conn.cursor()

@contextmanager
def chunk_transaction(conn: sqlite3.Connection, chunk_id: str) -> Generator[sqlite3.Cursor, None, None]:
    """
    Context manager for safe chunk-related database transactions.
    
    Args:
        conn: SQLite database connection
        chunk_id: ID of the chunk being processed
        
    Yields:
        SQLite cursor for database operations
        
    Raises:
        Exception: If any database operation fails
    """
    # Ensure fresh schema before transaction
    refresh_schema(conn)
    
    cursor = conn.cursor()
    try:
        # Start transaction with IMMEDIATE to prevent write conflicts
        cursor.execute("BEGIN IMMEDIATE")
        
        # Verify schema is accessible
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='chunks'
        """)
        if not cursor.fetchone():
            raise Exception("Chunks table not found - schema not accessible")
        
        yield cursor
        conn.commit()
        
        # Refresh schema after commit to ensure visibility of any changes
        refresh_schema(conn)
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction failed for chunk {chunk_id}: {e}")
        raise
    finally:
        cursor.close()

def verify_chunk_state(conn: sqlite3.Connection, chunk_id: str) -> bool:
    """
    Verify chunk state consistency.
    
    Args:
        conn: SQLite database connection
        chunk_id: ID of the chunk to verify
        
    Returns:
        bool: True if state is consistent, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        # Ensure schema is fresh before verification
        refresh_schema(conn)
        
        # First check if chunk exists
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", (chunk_id,))
        if cursor.fetchone()[0] == 0:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
            
        # Get current state
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id, embedding
            FROM chunks WHERE id = ?
        """, (chunk_id,))
        state = dict(zip(['embedding_status', 'qdrant_status', 'qdrant_id', 'embedding'], cursor.fetchone()))
        
        # Check for inconsistent states
        is_inconsistent = False
        
        # Check embedding status consistency
        if state['embedding_status'] == 'completed':
            if state['embedding'] is None:
                logger.warning(f"Chunk {chunk_id}: embedding_status completed but no embedding")
                is_inconsistent = True
        
        # Check Qdrant status consistency
        if state['qdrant_status'] == 'completed':
            if not state['qdrant_id']:
                logger.warning(f"Chunk {chunk_id}: qdrant_status completed but no qdrant_id")
                is_inconsistent = True
            if state['embedding_status'] != 'completed':
                logger.warning(f"Chunk {chunk_id}: qdrant_status completed but embedding_status not completed")
                is_inconsistent = True
            
        if is_inconsistent:
            logger.warning(f"Inconsistent state detected for chunk {chunk_id}: {dict(state)}")
            
        return not is_inconsistent
    finally:
        cursor.close()

def update_chunk_status(
    conn: sqlite3.Connection,
    chunk_id: str,
    status: str,
    qdrant_id: Optional[str] = None,
    embedding: Optional[bytes] = None,
    embedding_status: Optional[str] = None,
    qdrant_status: Optional[str] = None,
    error: Optional[str] = None,
    last_verified_at: Optional[datetime] = None
) -> None:
    """
    Update chunk status with proper transaction handling and state verification.
    
    Args:
        conn: SQLite database connection
        chunk_id: ID of the chunk to update
        status: New status to set ('pending', 'processing', 'completed', 'failed')
        qdrant_id: Optional Qdrant point ID
        embedding: Optional embedding data
        embedding_status: Optional embedding processing status
        qdrant_status: Optional Qdrant upload status
        error: Optional error message
        last_verified_at: Optional verification timestamp
        
    Raises:
        Exception: If update fails or results in inconsistent state
    """
    with chunk_transaction(conn, chunk_id) as cursor:
        try:
            # Log before state
            cursor.execute("""
                SELECT embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
            """, (chunk_id,))
            before_state = cursor.fetchone()
            logger.debug(f"Before update - Chunk {chunk_id}: {before_state}")
            
            # Simple, direct updates for better reliability
            cursor.execute("""
                UPDATE chunks 
                SET embedding_status = ?,
                    qdrant_status = ?,
                    qdrant_id = ?,
                    embedding = ?,
                    error_message = ?,
                    last_verified_at = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                embedding_status or (status if status in ['completed', 'failed'] else None),
                qdrant_status or status,
                qdrant_id,
                embedding,
                error,
                last_verified_at.isoformat() if last_verified_at else None,
                chunk_id
            ))
            
            # Log after state
            cursor.execute("""
                SELECT embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
            """, (chunk_id,))
            after_state = cursor.fetchone()
            logger.debug(f"After update - Chunk {chunk_id}: {after_state}")
            
            # Verify state consistency immediately
            if not verify_chunk_state(conn, chunk_id):
                raise Exception(f"Inconsistent state detected after update for chunk {chunk_id}")
            
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            raise