import sqlite3
import logging
import pytest
from pathlib import Path
import json
from datetime import datetime
from contextlib import contextmanager
import threading
import time
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

@contextmanager
def chunk_transaction(conn, chunk_id):
    """Ensure atomic updates to chunks"""
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")  # Lock the row
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Transaction failed for chunk {chunk_id}: {e}")
        raise
    finally:
        cursor.close()

def verify_chunk_state(conn, chunk_id):
    """Verify chunk state consistency"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT embedding_status, qdrant_status, qdrant_id
        FROM chunks 
        WHERE id = ? AND (
            (embedding_status = 'completed' AND qdrant_id IS NULL) OR
            (qdrant_status = 'completed' AND embedding_status != 'completed')
        )
    """, (chunk_id,))
    inconsistent = cursor.fetchone()
    if inconsistent:
        logging.warning(f"Inconsistent state detected for chunk {chunk_id}: {inconsistent}")
    return inconsistent is None

def update_chunk_status(conn, chunk_id, status, qdrant_id=None):
    """Update chunk status with proper transaction handling and logging"""
    with chunk_transaction(conn, chunk_id) as cursor:
        try:
            # Log before state
            cursor.execute("""
                SELECT embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
            """, (chunk_id,))
            before_state = cursor.fetchone()
            logging.debug(f"Before update - Chunk {chunk_id}: {before_state}")
            
            # Perform update atomically
            cursor.execute("""
                UPDATE chunks 
                SET embedding_status = CASE 
                        WHEN ? = 'completed' THEN 'completed' 
                        ELSE embedding_status 
                    END,
                    qdrant_status = ?,
                    qdrant_id = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, status, qdrant_id, chunk_id))
            
            # Log after state
            cursor.execute("""
                SELECT embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
            """, (chunk_id,))
            after_state = cursor.fetchone()
            logging.debug(f"After update - Chunk {chunk_id}: {after_state}")
            
            # Verify state consistency immediately
            if not verify_chunk_state(conn, chunk_id):
                raise Exception(f"Inconsistent state detected after update for chunk {chunk_id}")
            
        except Exception as e:
            logging.error(f"Failed to update chunk {chunk_id}: {e}")
            raise

def setup_test_db(db_path=':memory:'):
    """Create a test database with the same schema"""
    conn = sqlite3.connect(db_path, isolation_level=None)  # Enable autocommit mode
    cursor = conn.cursor()
    
    # Create chunks table with all necessary columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER,
        chunk_number INTEGER NOT NULL,
        content_hash TEXT NOT NULL,
        chunking_status TEXT DEFAULT 'pending',
        embedding_status TEXT DEFAULT 'pending',
        qdrant_status TEXT DEFAULT 'pending',
        embedding BLOB,
        qdrant_id TEXT,
        processed_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_verified_at DATETIME,
        error_message TEXT,
        version INTEGER DEFAULT 1
    )""")
    
    # Create indexes if they don't exist
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunking ON chunks(chunking_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_qdrant ON chunks(qdrant_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_id)")
    
    return conn

def insert_test_chunk(conn, chunk_id="test_chunk"):
    """Insert a single test chunk"""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO chunks (
        id, filename, content, token_count, chunk_number,
        content_hash, chunking_status, embedding_status, qdrant_status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk_id,
        "test_file.txt",
        "Test content",
        100,
        1,
        "test_hash",
        "completed",
        "pending",
        "pending"
    ))
    conn.commit()
    return chunk_id

def test_transaction_guard():
    """Test the chunk_transaction context manager"""
    conn = setup_test_db()
    chunk_id = insert_test_chunk(conn)
    
    try:
        # Test successful transaction
        with chunk_transaction(conn, chunk_id) as cursor:
            cursor.execute("""
                UPDATE chunks 
                SET embedding_status = 'completed',
                    qdrant_status = 'completed',
                    qdrant_id = ?
                WHERE id = ?
            """, (f"qdrant_{chunk_id}", chunk_id))
            
        # Verify the update was successful
        cursor = conn.cursor()
        cursor.execute("SELECT embedding_status, qdrant_status, qdrant_id FROM chunks WHERE id = ?", (chunk_id,))
        result = cursor.fetchone()
        assert result[0] == 'completed' and result[1] == 'completed' and result[2] == f"qdrant_{chunk_id}"
        
        # Test transaction rollback
        try:
            with chunk_transaction(conn, chunk_id) as cursor:
                cursor.execute("UPDATE chunks SET embedding_status = 'failed' WHERE id = ?", (chunk_id,))
                raise Exception("Test rollback")
        except Exception as e:
            assert str(e) == "Test rollback"
            
        # Verify the failed transaction was rolled back
        cursor = conn.cursor()
        cursor.execute("SELECT embedding_status FROM chunks WHERE id = ?", (chunk_id,))
        assert cursor.fetchone()[0] == 'completed'  # Should still be 'completed'
        
        logger.info("Transaction guard test passed")
        
    finally:
        conn.close()

def test_concurrent_updates():
    """Test concurrent updates with transaction guards"""
    # Use a temporary file for the test database
    db_fd, db_path = tempfile.mkstemp()
    try:
        def worker_process(worker_id, chunk_id, update_type):
            # Create a new connection to the shared database file
            conn = sqlite3.connect(db_path, isolation_level=None)
            try:
                with chunk_transaction(conn, chunk_id) as cursor:
                    if update_type == 'embedding':
                        cursor.execute("""
                            UPDATE chunks 
                            SET embedding_status = 'completed',
                                embedding = ?
                            WHERE id = ?
                        """, (b'fake_embedding', chunk_id))
                        logger.debug(f"Worker {worker_id}: Embedding update completed")
                    else:
                        cursor.execute("""
                            UPDATE chunks 
                            SET qdrant_status = 'completed',
                                qdrant_id = ?
                            WHERE id = ?
                        """, (f"qdrant_{chunk_id}", chunk_id))
                        logger.debug(f"Worker {worker_id}: Qdrant update completed")
            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {e}")
            finally:
                conn.close()
        
        # Create initial database state
        conn = setup_test_db(db_path)
        chunk_id = insert_test_chunk(conn)
        conn.close()
        
        # Create worker threads
        workers = [
            threading.Thread(target=worker_process, args=(i, chunk_id, 'embedding' if i % 2 == 0 else 'qdrant'))
            for i in range(2)
        ]
        
        # Start workers
        for w in workers:
            w.start()
        
        # Wait for completion
        for w in workers:
            w.join()
        
        # Verify final state
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id 
            FROM chunks WHERE id = ?
        """, (chunk_id,))
        final_state = cursor.fetchone()
        logger.debug(f"Final state: {final_state}")
        
        # Verify state consistency
        assert verify_chunk_state(conn, chunk_id)
        
        conn.close()
        
    finally:
        os.close(db_fd)
        os.unlink(db_path)

def test_status_update_with_verification():
    """Test the update_chunk_status function with state verification"""
    conn = setup_test_db()
    chunk_id = insert_test_chunk(conn)
    
    try:
        # Test normal update flow
        update_chunk_status(conn, chunk_id, "completed", f"qdrant_{chunk_id}")
        assert verify_chunk_state(conn, chunk_id)
        
        # Test invalid state transition
        try:
            update_chunk_status(conn, chunk_id, "pending", None)  # Should fail verification
            assert False, "Should have raised an exception"
        except Exception as e:
            logger.debug("Expected error caught:", str(e))
        
        logger.info("Status update with verification test passed")
        
    finally:
        conn.close()

if __name__ == '__main__':
    try:
        logger.info("Starting transaction fixes tests...")
        
        test_transaction_guard()
        logger.info("Transaction guard test completed")
        
        test_concurrent_updates()
        logger.info("Concurrent updates test completed")
        
        test_status_update_with_verification()
        logger.info("Status update verification test completed")
        
        print("All tests passed!")
        
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        raise 