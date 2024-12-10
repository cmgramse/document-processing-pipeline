"""Test document management functionality."""
import pytest
from datetime import datetime
from src.management.document_manager import DocumentManager

def test_check_document_exists(temp_db, sample_processed_doc):
    """Test document existence checking."""
    manager = DocumentManager(temp_db)
    
    # Test existing document
    exists, status = manager.check_document_exists(sample_processed_doc)
    assert exists
    assert status == "embedded"
    
    # Test non-existent document
    exists, status = manager.check_document_exists("nonexistent.md")
    assert not exists
    assert status is None

def test_delete_document(temp_db, sample_processed_doc, mock_qdrant_client):
    """Test document deletion."""
    manager = DocumentManager(temp_db, mock_qdrant_client)
    
    # Test successful deletion
    assert manager.delete_document(sample_processed_doc)
    
    # Verify deletion
    c = temp_db.cursor()
    c.execute("SELECT * FROM processed_files WHERE filename = ?", (sample_processed_doc,))
    assert not c.fetchone()
    c.execute("SELECT * FROM chunks WHERE filename = ?", (sample_processed_doc,))
    assert not c.fetchone()
    c.execute("SELECT * FROM documents WHERE filename = ?", (sample_processed_doc,))
    assert not c.fetchone()
    
    # Verify Qdrant deletion was called
    mock_qdrant_client.delete_vectors_by_filter.assert_called_once()

def test_select_documents():
    """Test document selection functionality."""
    manager = DocumentManager(None)  # Connection not needed for selection
    available_docs = ["doc1.md", "doc2.md", "doc3.md", "doc4.md", "doc5.md"]
    
    # Test individual selection
    selected = manager.select_documents(available_docs, "1,3,5")
    assert selected == ["doc1.md", "doc3.md", "doc5.md"]
    
    # Test range selection
    selected = manager.select_documents(available_docs, "2-4")
    assert selected == ["doc2.md", "doc3.md", "doc4.md"]
    
    # Test combination
    selected = manager.select_documents(available_docs, "1,3-5")
    assert selected == ["doc1.md", "doc3.md", "doc4.md", "doc5.md"]
    
    # Test 'all' selection
    selected = manager.select_documents(available_docs, "all")
    assert selected == available_docs
    
    # Test 'latest' selection
    selected = manager.select_documents(available_docs, "latest:3")
    assert selected == ["doc3.md", "doc4.md", "doc5.md"]
    
    # Test invalid selection
    selected = manager.select_documents(available_docs, "invalid")
    assert selected == []

def test_get_document_stats(temp_db, sample_processed_doc):
    """Test document statistics retrieval."""
    manager = DocumentManager(temp_db)
    stats = manager.get_document_stats(sample_processed_doc)
    
    assert stats["status"] == "embedded"
    assert stats["chunk_count"] == 2
    assert stats["total_chunks"] == 2
    assert stats["completed_chunks"] == 2
    assert stats["failed_chunks"] == 0
    assert stats["total_vectors"] == 2
    assert stats["uploaded_vectors"] == 2

def test_handle_existing_document(temp_db, sample_processed_doc, monkeypatch):
    """Test handling of existing documents."""
    manager = DocumentManager(temp_db)
    
    # Mock user input for different scenarios
    def mock_input(prompt):
        return "1"  # Choose reprocess
    monkeypatch.setattr('builtins.input', mock_input)
    
    # Test reprocess choice
    assert manager.handle_existing_document(sample_processed_doc)
    
    # Test non-existent document
    assert manager.handle_existing_document("nonexistent.md")

@pytest.mark.parametrize("test_input,expected", [
    ("1,2,3", 3),  # Individual numbers
    ("1-3", 3),    # Range
    ("1,3-5", 4),  # Combination
    ("all", 5),    # All documents
    ("latest:3", 3)  # Latest N
])
def test_batch_process_documents_selection(test_input, expected):
    """Test different document selection patterns."""
    manager = DocumentManager(None)
    available_docs = [f"doc{i}.md" for i in range(1, 6)]
    
    def mock_input(prompt):
        return test_input
    
    selected = manager.select_documents(available_docs, test_input)
    assert len(selected) == expected
