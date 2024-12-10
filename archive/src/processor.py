"""
Document processing pipeline module.

This module handles the processing of documents through various stages:
chunking, embedding generation, and Qdrant storage.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any
from src.database.transaction import chunk_transaction, update_chunk_status, verify_chunk_state

logger = logging.getLogger(__name__)

def process_chunk_batch(chunks: List[Dict[str, Any]], conn: sqlite3.Connection) -> None:
    """
    Process a batch of chunks through the pipeline.
    
    Args:
        chunks: List of chunks to process
        conn: SQLite database connection
    """
    try:
        # Process embeddings
        embeddings = generate_embeddings([chunk['content'] for chunk in chunks])
        
        # Upload to Qdrant
        qdrant_ids = upload_to_qdrant(embeddings)
        
        # Update database with results - using transaction for each chunk
        for chunk, embedding, qdrant_id in zip(chunks, embeddings, qdrant_ids):
            try:
                with chunk_transaction(conn, chunk['id']) as cursor:
                    # First update embedding
                    cursor.execute("""
                        UPDATE chunks 
                        SET embedding_status = 'completed',
                            embedding = ?
                        WHERE id = ?
                    """, (embedding, chunk['id']))
                    
                    # Then update qdrant status
                    cursor.execute("""
                        UPDATE chunks 
                        SET qdrant_status = 'completed',
                            qdrant_id = ?,
                            processed_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (qdrant_id, chunk['id']))
                    
                logger.debug(f"Successfully processed chunk {chunk['id']}")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk['id']}: {e}")
                with chunk_transaction(conn, chunk['id']) as cursor:
                    cursor.execute("""
                        UPDATE chunks 
                        SET embedding_status = 'failed',
                            qdrant_status = 'failed',
                            error_message = ?
                        WHERE id = ?
                    """, (str(e), chunk['id']))
                
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        # Mark all chunks as failed
        for chunk in chunks:
            try:
                with chunk_transaction(conn, chunk['id']) as cursor:
                    cursor.execute("""
                        UPDATE chunks 
                        SET embedding_status = 'failed',
                            qdrant_status = 'failed',
                            error_message = ?
                        WHERE id = ?
                    """, (str(e), chunk['id']))
            except Exception as update_error:
                logger.error(f"Failed to update failure status for chunk {chunk['id']}: {update_error}")

def process_pending_chunks(conn: sqlite3.Connection, batch_size: int = 50) -> Tuple[int, int]:
    """
    Process all pending chunks in the database.
    
    Args:
        conn: SQLite database connection
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    processed_count = 0
    error_count = 0
    
    try:
        while True:
            # Get next batch of pending chunks
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, filename
                FROM chunks
                WHERE embedding_status = 'pending'
                   OR (embedding_status = 'failed' AND error_count < 3)
                LIMIT ?
            """, (batch_size,))
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'content': row[1],
                    'filename': row[2]
                })
            
            if not chunks:
                break
                
            try:
                process_chunk_batch(chunks, conn)
                processed_count += len(chunks)
            except Exception as e:
                logger.error(f"Error processing chunk batch: {e}")
                error_count += len(chunks)
                
    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        raise
        
    return processed_count, error_count