"""Test database maintenance functionality."""
import pytest
from datetime import datetime, timedelta
import sqlite3
from src.database.maintenance import cleanup_database, track_chunk_versions, optimize_batch_processing

def test_cleanup_old_chunks(temp_db):
    """Test cleanup of old processed chunks."""
    c = temp_db.cursor()
    
    # Create old and new chunks
    old_date = (datetime.now() - timedelta(days=40)).isoformat()
    new_date = datetime.now().isoformat()
    
    # Insert test data
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("old_chunk", "test.md", "content", "completed", old_date))
    
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("new_chunk", "test.md", "content", "completed", new_date))
    
    c.execute("""
        INSERT INTO documents (id, chunk_id, filename, qdrant_status)
        VALUES (?, ?, ?, ?)
    """, ("doc1", "old_chunk", "test.md", "uploaded"))
    
    temp_db.commit()
    
    # Run cleanup
    stats = cleanup_database(temp_db, retention_days=30)
    
    # Verify old chunk was removed
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("old_chunk",))
    assert c.fetchone()[0] == 0
    
    # Verify new chunk remains
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("new_chunk",))
    assert c.fetchone()[0] == 1
    
    assert stats["chunks_removed"] == 1

def test_cleanup_failed_chunks(temp_db):
    """Test cleanup of repeatedly failed chunks."""
    c = temp_db.cursor()
    
    # Insert multiple failed attempts
    for i in range(3):
        c.execute("""
            INSERT INTO chunks (id, filename, chunk_number, content, embedding_status)
            VALUES (?, ?, ?, ?, ?)
        """, (f"failed_{i}", "test.md", 1, "content", "failed"))
    
    temp_db.commit()
    
    # Run cleanup
    stats = cleanup_database(temp_db)
    
    # Verify chunks were removed
    c.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status = ?", ("failed",))
    assert c.fetchone()[0] == 0
    
    assert stats["failed_chunks_removed"] > 0

def test_chunk_version_tracking(temp_db):
    """Test chunk version tracking functionality."""
    track_chunk_versions(temp_db)
    c = temp_db.cursor()
    
    # Insert test chunks
    c.execute("""
        INSERT INTO chunks (id, filename, content)
        VALUES (?, ?, ?)
    """, ("chunk1", "test.md", "content1"))
    
    c.execute("""
        INSERT INTO chunks (id, filename, content)
        VALUES (?, ?, ?)
    """, ("chunk2", "test.md", "content1"))  # Same content
    
    temp_db.commit()
    
    # Verify content hashes
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk1",))
    hash1 = c.fetchone()[0]
    
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk2",))
    hash2 = c.fetchone()[0]
    
    # Same content should have same hash
    assert hash1 == hash2
    
    # Different content should have different hash
    c.execute("""
        INSERT INTO chunks (id, filename, content)
        VALUES (?, ?, ?)
    """, ("chunk3", "test.md", "different content"))
    
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk3",))
    hash3 = c.fetchone()[0]
    
    assert hash1 != hash3

def test_batch_optimization():
    """Test batch processing optimization."""
    chunks = ["short text"] * 100
    batch_size = 10
    
    # Test batch size limit
    batches = list(optimize_batch_processing(chunks, batch_size=batch_size))
    assert all(len(batch) <= batch_size for batch in batches)
    
    # Test token limit
    long_chunks = ["very long text " * 100] * 20
    batches = list(optimize_batch_processing(long_chunks, batch_size=10))
    
    total_tokens = sum(len(chunk.split()) for chunk in long_chunks)
    batch_tokens = [sum(len(chunk.split()) for chunk in batch) for batch in batches]
    
    # Each batch should respect token limit
    assert all(tokens <= 8000 for tokens in batch_tokens)
    # All chunks should be processed
    assert sum(len(batch) for batch in batches) == len(long_chunks)

def test_cleanup_transaction_safety(temp_db):
    """Test transaction safety during cleanup."""
    c = temp_db.cursor()
    
    # Create test data
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status)
        VALUES (?, ?, ?, ?)
    """, ("test_chunk", "test.md", "content", "completed"))
    
    temp_db.commit()
    
    # Force an error during cleanup
    def failing_delete(*args):
        raise sqlite3.Error("Simulated error")
    
    original_execute = c.execute
    c.execute = failing_delete
    
    # Cleanup should rollback on error
    with pytest.raises(Exception):
        cleanup_database(temp_db)
    
    # Restore original execute
    c.execute = original_execute
    
    # Verify data still exists
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("test_chunk",))
    assert c.fetchone()[0] == 1
