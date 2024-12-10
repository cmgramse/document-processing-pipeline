"""
Test script to verify schema visibility issues in SQLite WAL mode.
"""

import sqlite3
import logging
import os
from datetime import datetime
from src.database.transaction import update_chunk_status, refresh_schema

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def setup_test_db():
    """Create a test database with schema"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create chunks table
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
    
    conn.commit()
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

def test_schema_visibility():
    """Test schema visibility and state consistency across transactions"""
    logging.info("Running schema visibility tests...")
    
    # Setup test database
    conn = setup_test_db()
    
    try:
        # Insert test chunk
        chunk_id = insert_test_chunk(conn)
        
        # Create a mock embedding
        mock_embedding = b'mock_embedding_data'
        
        # Update chunk status with embedding
        update_chunk_status(
            conn=conn,
            chunk_id=chunk_id,
            status='completed',
            embedding_status='completed',
            qdrant_status='completed',
            qdrant_id='qdrant_123',
            embedding=mock_embedding
        )
        
        # Verify final state
        cursor = conn.cursor()
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id, embedding
            FROM chunks WHERE id = ?
        """, (chunk_id,))
        final_state = cursor.fetchone()
        
        assert final_state[0] == 'completed', "Embedding status should be completed"
        assert final_state[1] == 'completed', "Qdrant status should be completed"
        assert final_state[2] == 'qdrant_123', "Qdrant ID should be set"
        assert final_state[3] == mock_embedding, "Embedding should be set"
        
        logging.info("Tests passed successfully")
        
    except Exception as e:
        logging.error(f"Tests failed: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    test_schema_visibility()