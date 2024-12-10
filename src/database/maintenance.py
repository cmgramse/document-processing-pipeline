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
import json
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3

logger = logging.getLogger(__name__)

def cleanup_database(conn, retention_days: int = 30) -> Dict[str, int]:
    """
    Comprehensive cleanup of database and vector store.
    
    Args:
        conn: Database connection
        retention_days: Days to retain processed data
        
    Returns:
        Cleanup statistics
    """
    stats = {
        'chunks_deleted': 0,
        'vectors_deleted': 0,
        'orphaned_vectors': 0,
        'errors': 0
    }
    
    try:
        # Start transaction
        c = conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        # 1. Find old chunks to delete
        old_chunks = c.execute("""
            SELECT c.id, c.qdrant_id 
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.created_at < ? AND (
                d.id IS NULL OR  -- Orphaned chunks
                d.status = 'deleted'  -- Chunks from deleted documents
            )
        """, (cutoff_date,)).fetchall()
        
        if old_chunks:
            # 2. Delete vectors from Qdrant
            qdrant_ids = [chunk['qdrant_id'] for chunk in old_chunks if chunk['qdrant_id']]
            if qdrant_ids:
                deleted = delete_vectors_from_qdrant(qdrant_ids)
                stats['vectors_deleted'] = deleted
            
            # 3. Delete chunks from database
            chunk_ids = [chunk['id'] for chunk in old_chunks]
            c.executemany("""
                DELETE FROM chunks WHERE id = ?
            """, [(id,) for id in chunk_ids])
            stats['chunks_deleted'] = len(chunk_ids)
            
        # 4. Find and cleanup orphaned vectors
        orphaned = find_orphaned_vectors(conn)
        if orphaned:
            deleted = delete_vectors_from_qdrant(orphaned)
            stats['orphaned_vectors'] = deleted
        
        # 5. Cleanup processing history
        c.execute("""
            DELETE FROM processing_history 
            WHERE created_at < ?
        """, (cutoff_date,))
        
        # 6. Cleanup processing queue
        c.execute("""
            DELETE FROM processing_queue
            WHERE (status = 'completed' OR status = 'failed')
            AND created_at < ?
        """, (cutoff_date,))
        
        # 7. Vacuum database
        c.execute("VACUUM")
        
        conn.commit()
        logger.info(f"Cleanup completed successfully: {json.dumps(stats)}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during cleanup: {str(e)}")
        stats['errors'] += 1
        
    return stats

def find_orphaned_vectors(conn) -> List[str]:
    """Find vectors in Qdrant that don't have corresponding chunks."""
    try:
        # Get all Qdrant IDs from database
        c = conn.cursor()
        db_vectors = set(row[0] for row in c.execute(
            "SELECT DISTINCT qdrant_id FROM chunks WHERE qdrant_id IS NOT NULL"
        ).fetchall())
        
        # Get all vectors from Qdrant
        qdrant_vectors = get_all_qdrant_vectors()
        
        # Find orphaned vectors
        orphaned = list(set(qdrant_vectors) - db_vectors)
        if orphaned:
            logger.info(f"Found {len(orphaned)} orphaned vectors")
            
        return orphaned
        
    except Exception as e:
        logger.error(f"Error finding orphaned vectors: {str(e)}")
        return []

def delete_vectors_from_qdrant(vector_ids: List[str]) -> int:
    """Delete vectors from Qdrant collection."""
    if not vector_ids:
        return 0
        
    try:
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        qdrant_url = os.environ["QDRANT_URL"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Delete in batches
        batch_size = 100
        deleted = 0
        
        for i in range(0, len(vector_ids), batch_size):
            batch = vector_ids[i:i + batch_size]
            
            response = requests.post(
                f"{qdrant_url}/collections/{collection_name}/points/delete",
                headers=headers,
                json={"points": batch}
            )
            
            if response.status_code == 200:
                deleted += len(batch)
                logger.info(f"Deleted batch of {len(batch)} vectors")
            else:
                logger.error(
                    f"Failed to delete vectors: {response.status_code} - {response.text}"
                )
        
        return deleted
        
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        return 0

def get_all_qdrant_vectors() -> List[str]:
    """Get all vector IDs from Qdrant collection."""
    try:
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        qdrant_url = os.environ["QDRANT_URL"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Get collection info first
        response = requests.get(
            f"{qdrant_url}/collections/{collection_name}",
            headers=headers
        )
        response.raise_for_status()
        
        total_vectors = response.json()['result']['vectors_count']
        
        # Scroll through all vectors
        vector_ids = []
        offset = None
        limit = 100
        
        while len(vector_ids) < total_vectors:
            payload = {
                "limit": limit,
                "with_payload": False,
                "with_vector": False
            }
            if offset:
                payload["offset"] = offset
            
            response = requests.post(
                f"{qdrant_url}/collections/{collection_name}/points/scroll",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()['result']
            vector_ids.extend(point['id'] for point in result['points'])
            
            if not result.get('next_page_offset'):
                break
                
            offset = result['next_page_offset']
        
        return vector_ids
        
    except Exception as e:
        logger.error(f"Error getting Qdrant vectors: {str(e)}")
        return []

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
    """Track and manage chunk versions in the database."""
    c = conn.cursor()
    stats = {'current_version': 0, 'chunks_updated': 0, 'errors': 0}
    
    try:
        # Check if version column exists
        c.execute("PRAGMA table_info(chunks)")
        columns = [col[1] for col in c.fetchall()]
        if 'version' not in columns:
            # Add version column if it doesn't exist
            c.execute('ALTER TABLE chunks ADD COLUMN version INTEGER DEFAULT 1')
            conn.commit()
            stats['chunks_updated'] = c.rowcount
            return stats
        
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
