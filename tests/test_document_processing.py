"""Test document processing functionality."""
import pytest
from datetime import datetime
from pathlib import Path
from src.processing.documents import process_documents
from src.database.maintenance import optimize_batch_processing
from src.api.jina import segment_text, get_embeddings

def test_process_new_document(temp_db, test_docs_dir, mock_jina_api):
    """Test processing of a new document."""
    test_file = test_docs_dir / "test1.md"
    documents, stats = process_documents([str(test_file)], temp_db)
    
    # Debug: Check database state
    c = temp_db.cursor()
    
    print("\nChunks table:")
    c.execute("SELECT * FROM chunks")
    chunks = c.fetchall()
    for chunk in chunks:
        print(f"Chunk: {chunk}")
    
    print("\nDocuments table:")
    c.execute("SELECT * FROM documents")
    docs = c.fetchall()
    for doc in docs:
        print(f"Doc: {doc}")
    
    print("\nProcessed documents:")
    for doc in documents:
        print(f"Processed doc: {doc}")
    
    assert len(documents) == 2  # Two chunks from mock
    assert stats["chunks_created"] == 2
    assert stats["embeddings_generated"] == 2
    
    # Check database entries
    c = temp_db.cursor()
    
    # Check processed_files
    c.execute("SELECT status, chunk_count FROM processed_files WHERE filename = ?", (test_file.name,))
    result = c.fetchone()
    assert result
    assert result[0] == "embedded"
    assert result[1] == 2
    
    # Check chunks
    c.execute("SELECT COUNT(*) FROM chunks WHERE filename = ?", (test_file.name,))
    assert c.fetchone()[0] == 2
    
    # Check documents
    c.execute("SELECT COUNT(*) FROM documents WHERE filename = ?", (test_file.name,))
    assert c.fetchone()[0] == 2

def test_force_reprocess_document(temp_db, test_docs_dir, mock_jina_api, sample_processed_doc):
    """Test force reprocessing of an existing document."""
    test_file = test_docs_dir / sample_processed_doc
    
    # Create test file
    test_file.write_text("Test document content")
    
    documents, stats = process_documents([str(test_file)], temp_db, force_reprocess=[str(test_file)])
    
    assert len(documents) == 2
    assert stats["chunks_created"] == 2
    assert stats["embeddings_generated"] == 2
    
    # Verify new processing timestamp
    c = temp_db.cursor()
    c.execute("SELECT processed_at FROM processed_files WHERE filename = ?", (sample_processed_doc,))
    new_timestamp = c.fetchone()[0]
    assert new_timestamp  # Should have a new timestamp

def test_batch_processing_optimization():
    """Test batch processing optimization."""
    # Test with short chunks
    chunks = ["short text", "medium text " * 5, "long text " * 10]
    batches = list(optimize_batch_processing(chunks))
    assert len(batches) > 0
    assert all(len(batch) <= 50 for batch in batches)

    # Test with custom batch size
    batches = list(optimize_batch_processing(chunks, batch_size=2))
    assert len(batches) >= 2
    assert all(len(batch) <= 2 for batch in batches)

    # Test with very long chunks that should trigger token limit splits
    long_chunks = ["very long text " * 500] * 5  # Much longer chunks
    batches = list(optimize_batch_processing(long_chunks))
    assert len(batches) > 1  # Should split due to token limit

def test_error_handling(temp_db, test_docs_dir, mock_jina_api, monkeypatch):
    """Test error handling during processing."""
    def mock_failing_post(*args, **kwargs):
        raise Exception("Embedding failed")
    
    monkeypatch.setattr("requests.post", mock_failing_post)
    
    test_file = test_docs_dir / "test1.md"
    test_file.write_text("Test content")
    
    documents, stats = process_documents([str(test_file)], temp_db)
    
    # Should have created chunks but failed embeddings
    assert stats["chunks_created"] > 0
    assert stats["embeddings_generated"] == 0
    
    # Check failure status in database
    c = temp_db.cursor()
    c.execute("SELECT embedding_status FROM chunks WHERE filename = ?", (test_file.name,))
    statuses = [row[0] for row in c.fetchall()]
    assert all(status == "failed" for status in statuses)

@pytest.mark.parametrize("content,expected_chunks", [
    ("Short text", 1),
    ("Medium text\nWith multiple\nLines", 2),
    ("Very long text " * 20, 3)
])
def test_document_segmentation(temp_db, test_docs_dir, mock_jina_api, content, expected_chunks):
    """Test document segmentation with different content types."""
    # Create test file
    test_file = test_docs_dir / "test_segment.md"
    with open(test_file, 'w') as f:
        f.write(content)

    documents, stats = process_documents([str(test_file)], temp_db)
    assert stats["chunks_created"] == expected_chunks
