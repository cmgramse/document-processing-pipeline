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
    
    # Insert test data - one old chunk with no document reference
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("old_chunk_no_ref", "test.md", "content", "completed", old_date))
    
    # Insert another old chunk with document reference
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("old_chunk_with_ref", "test.md", "content", "completed", old_date))
    
    # Insert a new chunk
    c.execute("""
        INSERT INTO chunks (id, filename, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("new_chunk", "test.md", "content", "completed", new_date))
    
    # Insert document reference for one old chunk
    c.execute("""
        INSERT INTO documents (id, chunk_id, filename)
        VALUES (?, ?, ?)
    """, ("doc1", "old_chunk_with_ref", "test.md"))
    
    temp_db.commit()
    
    # Run cleanup
    stats = cleanup_database(temp_db, retention_days=30)
    
    # Verify old chunk without reference was removed
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("old_chunk_no_ref",))
    assert c.fetchone()[0] == 0
    
    # Verify old chunk with reference remains
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("old_chunk_with_ref",))
    assert c.fetchone()[0] == 1
    
    # Verify new chunk remains
    c.execute("SELECT COUNT(*) FROM chunks WHERE id = ?", ("new_chunk",))
    assert c.fetchone()[0] == 1
    
    assert stats["old_chunks_removed"] == 1

def test_cleanup_failed_chunks(temp_db):
    """Test cleanup of repeatedly failed chunks."""
    c = temp_db.cursor()
    
    # Insert multiple failed attempts with unique chunk numbers
    for i in range(3):
        c.execute("""
            INSERT INTO chunks (id, filename, chunk_number, content, embedding_status)
            VALUES (?, ?, ?, ?, ?)
        """, (f"failed_{i}", "test.md", i, "content", "failed"))
    
    temp_db.commit()
    
    # Run cleanup
    stats = cleanup_database(temp_db)
    
    # Verify cleanup
    c.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status = 'failed'")
    assert c.fetchone()[0] == 0
    
    assert stats["failed_chunks_removed"] > 0

def test_chunk_version_tracking(temp_db):
    """Test chunk version tracking functionality."""
    c = temp_db.cursor()
    
    # Insert test chunks with unique chunk numbers
    c.execute("""
        INSERT INTO chunks (id, filename, chunk_number, content, content_hash)
        VALUES (?, ?, ?, ?, ?)
    """, ("chunk1", "test.md", 1, "content1", hash("content1")))
    
    c.execute("""
        INSERT INTO chunks (id, filename, chunk_number, content, content_hash)
        VALUES (?, ?, ?, ?, ?)
    """, ("chunk2", "test.md", 2, "content1", hash("content1")))  # Same content
    
    c.execute("""
        INSERT INTO chunks (id, filename, chunk_number, content, content_hash)
        VALUES (?, ?, ?, ?, ?)
    """, ("chunk3", "test.md", 3, "different content", hash("different content")))
    
    temp_db.commit()
    
    # Verify content hashes
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk1",))
    hash1 = c.fetchone()[0]
    
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk2",))
    hash2 = c.fetchone()[0]
    
    # Same content should have same hash
    assert hash1 == hash2
    
    c.execute("SELECT content_hash FROM chunks WHERE id = ?", ("chunk3",))
    hash3 = c.fetchone()[0]
    
    # Different content should have different hash
    assert hash1 != hash3

def test_optimize_batch_processing(temp_db):
    """Test batch processing optimization."""
    # Test database optimization
    optimize_batch_processing(temp_db)

    # Verify pragmas were set
    c = temp_db.cursor()
    
    # Check journal mode
    c.execute('PRAGMA journal_mode')
    journal_mode = c.fetchone()[0].upper()
    assert journal_mode in ['MEMORY', 'DELETE'], f"Expected journal_mode to be MEMORY or DELETE, got {journal_mode}"
    
    # Check other pragmas with minimum requirements
    c.execute('PRAGMA synchronous')
    sync_mode = c.fetchone()[0]
    assert sync_mode <= 2, f"Expected synchronous <= 2 (NORMAL), got {sync_mode}"
    
    # temp_store can be 0 (default) or 2 (memory) depending on system configuration
    c.execute('PRAGMA temp_store')
    temp_store = c.fetchone()[0]
    assert temp_store in [0, 2], f"Expected temp_store to be 0 or 2, got {temp_store}"
    
    # Test list batching
    test_list = list(range(10))
    batches = list(optimize_batch_processing(test_list, batch_size=3))
    assert len(batches) == 4
    assert batches[0] == [0,1,2]
    assert batches[-1] == [9]

def test_cleanup_transaction_safety(temp_db):
    """Test transaction safety during cleanup."""
    c = temp_db.cursor()
    
    # Create a foreign key constraint that will cause the deletion to fail
    c.execute("""
        CREATE TABLE IF NOT EXISTS test_references (
            id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
    """)
    
    # Enable foreign key constraints
    c.execute("PRAGMA foreign_keys = ON")
    
    # Insert test data
    c.execute("""
        INSERT INTO chunks (id, filename, chunk_number, content, embedding_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("test_chunk", "test.md", 1, "content", "completed", 
          (datetime.now() - timedelta(days=40)).isoformat()))
    
    # Create a reference that will prevent deletion
    c.execute("""
        INSERT INTO test_references (id, chunk_id)
        VALUES (?, ?)
    """, ("ref1", "test_chunk"))
    
    temp_db.commit()
    
    # Run cleanup - should fail due to foreign key constraint
    try:
        cleanup_database(temp_db, retention_days=30)
        pytest.fail("Expected cleanup to fail due to foreign key constraint")
    except sqlite3.IntegrityError:
        pass  # This is expected
    
    # Verify data still exists
    c.execute("SELECT COUNT(*) FROM chunks")
    assert c.fetchone()[0] == 1
    
    # Clean up the test table
    c.execute("DROP TABLE test_references")
    temp_db.commit()
