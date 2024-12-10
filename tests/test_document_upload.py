"""
Test document upload process using pytest.
"""

import os
import logging
import pytest
from pathlib import Path
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.processing.documents import list_available_documents, select_documents
from src.pipeline.processor import process_document, process_pending_chunks, get_processing_stats
from src.api.jina import segment_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def required_env_vars():
    """Required environment variables."""
    return [
        'JINA_API_KEY',
        'JINA_EMBEDDING_MODEL',
        'QDRANT_API_KEY',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME'
    ]

@pytest.fixture
def sample_document(tmp_path):
    """Create a sample document for testing."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    doc_path = doc_dir / "test.txt"
    doc_path.write_text("This is a test document.\nIt has multiple lines.\nUsed for testing.")
    return str(doc_path)

@pytest.fixture
def mock_document_selection(monkeypatch):
    """Mock document selection to always select the first document."""
    def mock_select(*args, **kwargs):
        return [args[0][0][1]]  # Return first document's full path
    monkeypatch.setattr("src.processing.documents.select_documents", mock_select)

def test_environment_validation(required_env_vars):
    """Test environment variable validation."""
    # Temporarily set environment variables
    missing = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")

def test_document_listing(sample_document):
    """Test listing available documents."""
    available_docs = list_available_documents()
    assert available_docs, "No documents found"
    assert len(available_docs) >= 1, "Expected at least one document"
    assert any(doc[1] == sample_document for doc in available_docs), "Sample document not found"

def test_document_chunking(sample_document):
    """Test document chunking with Jina Segmenter."""
    with open(sample_document, 'r') as f:
        content = f.read()
    
    chunks = segment_text(content)
    assert chunks, "No chunks created"
    assert len(chunks) >= 1, "Expected at least one chunk"
    assert all(isinstance(chunk, str) for chunk in chunks), "Invalid chunk format"

def test_document_processing(sample_document):
    """Test full document processing pipeline."""
    with get_db() as session:
        # Read document
        with open(sample_document, 'r') as f:
            content = f.read()
        
        # Chunk document
        chunks = segment_text(content)
        token_counts = [len(chunk.split()) for chunk in chunks]
        
        # Process document
        success = process_document(session, sample_document, chunks, token_counts)
        assert success, "Document processing failed"
        
        # Process chunks
        stats = process_pending_chunks(session)
        assert stats['failed'] == 0, f"Chunk processing failed: {stats}"
        assert stats['processed'] > 0, "No chunks were processed"

def test_processing_stats(sample_document):
    """Test processing statistics after document upload."""
    with get_db() as session:
        stats = get_processing_stats(session)
        
        assert 'total_documents' in stats, "Missing document count"
        assert 'total_chunks' in stats, "Missing chunk count"
        assert 'completed_chunks' in stats, "Missing completed chunks count"
        assert 'failed_chunks' in stats, "Missing failed chunks count"
        
        # Verify counts
        assert stats['total_documents'] > 0, "No documents found"
        assert stats['total_chunks'] > 0, "No chunks found"
        assert stats['failed_chunks'] == 0, "Found failed chunks"

def test_full_pipeline(sample_document, mock_document_selection):
    """Test the complete document processing pipeline."""
    # List and select documents
    available_docs = list_available_documents()
    assert available_docs, "No documents found"
    
    selected_docs = select_documents(available_docs)
    assert selected_docs, "No documents selected"
    assert len(selected_docs) == 1, "Expected one document"
    
    # Process documents
    with get_db() as session:
        # Process each document
        for doc_path in selected_docs:
            # Read and chunk
            with open(doc_path, 'r') as f:
                content = f.read()
            chunks = segment_text(content)
            token_counts = [len(chunk.split()) for chunk in chunks]
            
            # Process
            success = process_document(session, doc_path, chunks, token_counts)
            assert success, "Document processing failed"
            
            # Process chunks
            stats = process_pending_chunks(session)
            assert stats['failed'] == 0, "Chunk processing failed"
            assert stats['processed'] > 0, "No chunks processed"
        
        # Verify final state
        stats = get_processing_stats(session)
        assert stats['completed_documents'] > 0, "No documents completed"
        assert stats['failed_documents'] == 0, "Found failed documents"
        assert stats['completed_chunks'] > 0, "No chunks completed"
        assert stats['failed_chunks'] == 0, "Found failed chunks" 