import sqlite3
import logging
import pytest
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_test_db():
    """Create a test database with the same schema"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create chunks table with all necessary columns
    cursor.execute("""
    CREATE TABLE chunks (
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
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_chunks_filename ON chunks(filename)")
    cursor.execute("CREATE INDEX idx_chunks_chunking ON chunks(chunking_status)")
    cursor.execute("CREATE INDEX idx_chunks_embedding ON chunks(embedding_status)")
    cursor.execute("CREATE INDEX idx_chunks_qdrant ON chunks(qdrant_status)")
    cursor.execute("CREATE INDEX idx_chunks_qdrant_id ON chunks(qdrant_id)")
    
    return conn

def insert_test_chunks(conn, num_chunks=5):
    """Insert test chunks into the database"""
    cursor = conn.cursor()
    chunks = []
    
    for i in range(num_chunks):
        chunk_id = f"test_chunk_{i}"
        chunk = {
            'id': chunk_id,
            'filename': f'test_file_{i//2}.txt',  # Group chunks by file
            'content': f'Test content {i}',
            'token_count': 100,
            'chunk_number': i,
            'content_hash': f'hash_{i}',
            'chunking_status': 'completed',
            'embedding_status': 'pending',
            'qdrant_status': 'pending'
        }
        cursor.execute("""
        INSERT INTO chunks (
            id, filename, content, token_count, chunk_number, 
            content_hash, chunking_status, embedding_status, qdrant_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk['id'], chunk['filename'], chunk['content'], 
            chunk['token_count'], chunk['chunk_number'], chunk['content_hash'],
            chunk['chunking_status'], chunk['embedding_status'], chunk['qdrant_status']
        ))
        chunks.append(chunk)
    
    conn.commit()
    return chunks

def simulate_chunk_processing(conn, chunk):
    """Simulate the chunk processing workflow"""
    cursor = conn.cursor()
    
    try:
        # Log initial state
        logger.debug(f"Processing chunk {chunk['id']}")
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk['id'],))
        initial_state = cursor.fetchone()
        logger.debug(f"Initial state: {dict(zip([col[0] for col in cursor.description], initial_state))}")
        
        # 1. Update embedding status
        logger.debug(f"Updating embedding status for chunk {chunk['id']}")
        cursor.execute("""
        UPDATE chunks 
        SET embedding_status = 'completed', 
            embedding = ? 
        WHERE id = ?
        """, (b'fake_embedding', chunk['id']))
        
        # Verify embedding update
        cursor.execute("SELECT embedding_status FROM chunks WHERE id = ?", (chunk['id'],))
        embedding_status = cursor.fetchone()[0]
        logger.debug(f"After embedding update - status: {embedding_status}")
        
        # 2. Update qdrant status and ID
        logger.debug(f"Updating qdrant status for chunk {chunk['id']}")
        cursor.execute("""
        UPDATE chunks 
        SET qdrant_status = 'completed',
            qdrant_id = ?,
            processed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """, (f"qdrant_{chunk['id']}", chunk['id']))
        
        # Verify qdrant update
        cursor.execute("SELECT qdrant_status, qdrant_id FROM chunks WHERE id = ?", (chunk['id'],))
        qdrant_result = cursor.fetchone()
        logger.debug(f"After qdrant update - status: {qdrant_result[0]}, id: {qdrant_result[1]}")
        
        conn.commit()
        logger.debug(f"Successfully processed chunk {chunk['id']}")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error processing chunk {chunk['id']}: {str(e)}")
        raise

def test_chunk_processing_workflow():
    """Test the entire chunk processing workflow"""
    conn = setup_test_db()
    chunks = insert_test_chunks(conn)
    
    try:
        # Process each chunk
        for chunk in chunks:
            # Start a new transaction
            with conn:
                cursor = conn.cursor()
                
                # Log chunk state before processing
                cursor.execute("""
                SELECT chunking_status, embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
                """, (chunk['id'],))
                before_state = cursor.fetchone()
                logger.debug(f"Before processing - Chunk {chunk['id']} state: {before_state}")
                
                # Process the chunk
                simulate_chunk_processing(conn, chunk)
                
                # Verify final state
                cursor.execute("""
                SELECT chunking_status, embedding_status, qdrant_status, qdrant_id 
                FROM chunks WHERE id = ?
                """, (chunk['id'],))
                after_state = cursor.fetchone()
                logger.debug(f"After processing - Chunk {chunk['id']} state: {after_state}")
                
                # Verify all status fields
                assert after_state[0] == 'completed', "Chunking status should be completed"
                assert after_state[1] == 'completed', "Embedding status should be completed"
                assert after_state[2] == 'completed', "Qdrant status should be completed"
                assert after_state[3] is not None, "Qdrant ID should be set"
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        conn.close()

def test_concurrent_chunk_updates():
    """Test concurrent updates to the same chunk"""
    conn1 = setup_test_db()
    conn2 = setup_test_db()  # Simulate second connection
    chunks = insert_test_chunks(conn1)
    
    try:
        # Start transaction in first connection
        cursor1 = conn1.cursor()
        cursor1.execute("BEGIN TRANSACTION")
        
        # Update chunk in first connection
        chunk = chunks[0]
        cursor1.execute("""
        UPDATE chunks 
        SET embedding_status = 'completed',
            embedding = ?
        WHERE id = ?
        """, (b'fake_embedding_1', chunk['id']))
        
        # Try to update same chunk in second connection
        cursor2 = conn2.cursor()
        cursor2.execute("BEGIN TRANSACTION")
        cursor2.execute("""
        UPDATE chunks 
        SET qdrant_status = 'completed',
            qdrant_id = ?
        WHERE id = ?
        """, (f"qdrant_{chunk['id']}", chunk['id']))
        
        # Commit both transactions
        conn1.commit()
        conn2.commit()
        
        # Verify final state
        cursor1.execute("""
        SELECT embedding_status, qdrant_status, qdrant_id 
        FROM chunks WHERE id = ?
        """, (chunk['id'],))
        final_state = cursor1.fetchone()
        logger.debug(f"Final chunk state: {final_state}")
        
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite operational error: {str(e)}")
        raise
    finally:
        conn1.close()
        conn2.close()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        test_chunk_processing_workflow()
        test_concurrent_chunk_updates()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {str(e)}")
