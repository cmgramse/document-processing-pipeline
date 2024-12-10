"""
Verification tests for chunk processing functionality.
"""

import sqlite3
import logging
import pytest
from pathlib import Path
import json
from datetime import datetime
import os
import tempfile
from typing import List, Dict, Any
import unittest.mock as mock
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_db():
    """Create a test database with the production schema"""
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

def insert_test_chunks(conn: sqlite3.Connection, num_chunks: int = 5) -> List[Dict[str, Any]]:
    """Insert test chunks into the database"""
    cursor = conn.cursor()
    chunks = []
    
    for i in range(num_chunks):
        chunk_id = f"test_chunk_{i}"
        chunk = {
            'id': chunk_id,
            'filename': f'test_file_{i//2}.txt',
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

def verify_chunk_states(conn: sqlite3.Connection, chunk_ids: List[str]) -> bool:
    """Verify the state consistency of chunks"""
    cursor = conn.cursor()
    
    for chunk_id in chunk_ids:
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id, error_message
            FROM chunks 
            WHERE id = ?
        """, (chunk_id,))
        state = cursor.fetchone()
        
        if not state:
            logger.error(f"Chunk {chunk_id} not found")
            return False
            
        embedding_status, qdrant_status, qdrant_id, error = state
        
        # Check for inconsistent states
        if qdrant_status == 'completed' and not qdrant_id:
            logger.error(f"Chunk {chunk_id} marked as completed but has no qdrant_id")
            return False
            
        if qdrant_status == 'completed' and embedding_status != 'completed':
            logger.error(f"Chunk {chunk_id} has completed qdrant but incomplete embedding")
            return False
            
        if error and embedding_status != 'failed' and qdrant_status != 'failed':
            logger.error(f"Chunk {chunk_id} has error but not marked as failed")
            return False
    
    return True

# Mock functions
def mock_generate_embeddings(texts):
    """Mock function for generating embeddings"""
    return [np.random.rand(1024).tobytes() for _ in texts]

def mock_upload_to_qdrant(embeddings):
    """Mock function for uploading to Qdrant"""
    return [f"qdrant_id_{i}" for i in range(len(embeddings))]

@mock.patch('src.pipeline.processor.generate_embeddings', side_effect=mock_generate_embeddings)
@mock.patch('src.pipeline.processor.upload_to_qdrant', side_effect=mock_upload_to_qdrant)
def test_single_chunk_processing(mock_upload, mock_embed):
    """Test processing of a single chunk"""
    from src.pipeline.processor import process_chunk_batch
    
    conn = setup_test_db()
    chunks = insert_test_chunks(conn, num_chunks=1)
    
    try:
        # Process the single chunk
        process_chunk_batch(chunks, conn)
        
        # Verify final state
        cursor = conn.cursor()
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id
            FROM chunks WHERE id = ?
        """, (chunks[0]['id'],))
        final_state = cursor.fetchone()
        
        assert final_state[0] == 'completed', "Embedding status should be completed"
        assert final_state[1] == 'completed', "Qdrant status should be completed"
        assert final_state[2] is not None, "Qdrant ID should be set"
        
        # Verify mock calls
        mock_embed.assert_called_once()
        mock_upload.assert_called_once()
        
        logger.info("Single chunk processing test passed")
        
    except Exception as e:
        logger.error(f"Single chunk test failed: {str(e)}", exc_info=True)
        raise
    finally:
        conn.close()

@mock.patch('src.pipeline.processor.generate_embeddings', side_effect=mock_generate_embeddings)
@mock.patch('src.pipeline.processor.upload_to_qdrant', side_effect=mock_upload_to_qdrant)
def test_batch_processing(mock_upload, mock_embed):
    """Test processing of multiple chunks in a batch"""
    from src.pipeline.processor import process_chunk_batch
    
    conn = setup_test_db()
    chunks = insert_test_chunks(conn, num_chunks=5)
    
    try:
        # Process the batch
        process_chunk_batch(chunks, conn)
        
        # Verify all chunks
        chunk_ids = [chunk['id'] for chunk in chunks]
        assert verify_chunk_states(conn, chunk_ids), "Chunk states are inconsistent"
        
        # Verify mock calls
        mock_embed.assert_called_once()
        mock_upload.assert_called_once()
        assert len(mock_embed.call_args[0][0]) == 5, "Should process 5 chunks"
        
        logger.info("Batch processing test passed")
        
    except Exception as e:
        logger.error(f"Batch processing test failed: {str(e)}", exc_info=True)
        raise
    finally:
        conn.close()

@mock.patch('src.pipeline.processor.generate_embeddings', side_effect=mock_generate_embeddings)
@mock.patch('src.pipeline.processor.upload_to_qdrant', side_effect=mock_upload_to_qdrant)
def test_error_handling(mock_upload, mock_embed):
    """Test error handling during processing"""
    from src.pipeline.processor import process_chunk_batch
    
    conn = setup_test_db()
    chunks = insert_test_chunks(conn, num_chunks=3)
    
    # Make the embedding generation fail for the second chunk
    def mock_generate_with_error(texts):
        if len(texts) > 1 and texts[1] is None:
            raise ValueError("Cannot process None content")
        return mock_generate_embeddings(texts)
    
    mock_embed.side_effect = mock_generate_with_error
    
    # Modify one chunk to cause an error
    chunks[1]['content'] = None
    
    try:
        # Process the batch
        process_chunk_batch(chunks, conn)
    except ValueError:
        pass  # We expect this error
        
    # Verify error handling
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, embedding_status, qdrant_status, error_message
        FROM chunks
        WHERE error_message IS NOT NULL
    """)
    failed_chunks = cursor.fetchall()
    
    assert len(failed_chunks) > 0, "No chunks marked as failed"
    for chunk in failed_chunks:
        assert chunk[1] == 'failed', "Failed chunk should have failed embedding status"
        assert chunk[2] == 'failed', "Failed chunk should have failed qdrant status"
        assert chunk[3] is not None, "Failed chunk should have error message"
    
    logger.info("Error handling test passed")
    conn.close()

if __name__ == '__main__':
    logger.info("Starting verification tests...")
    
    try:
        test_single_chunk_processing()
        test_batch_processing()
        test_error_handling()
        
        print("All verification tests passed!")
        
    except Exception as e:
        logger.error(f"Verification tests failed: {str(e)}", exc_info=True)
        raise 