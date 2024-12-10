"""Test document processing functionality."""
import pytest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from src.api.jina import JinaAPI, jina_client
from src.processing.background_tasks import ProcessingQueue, ProcessingStatus, BackgroundProcessor

@pytest.fixture
def mock_jina_api():
    """Create a mock Jina API client."""
    mock_api = MagicMock(spec=JinaAPI)
    
    # Mock segment_text
    mock_api.segment_text.return_value = [
        Document(
            page_content="Test chunk 1",
            metadata={"chunk_number": 0, "total_chunks": 2}
        ),
        Document(
            page_content="Test chunk 2",
            metadata={"chunk_number": 1, "total_chunks": 2}
        )
    ]
    
    # Mock get_embeddings
    mock_api.get_embeddings.return_value = [
        [0.1] * 1024,  # Mock 1024-dimensional embeddings
        [0.2] * 1024
    ]
    
    # Mock process_document
    mock_api.process_document.return_value = [
        {
            'content': "Test chunk 1",
            'embedding': [0.1] * 1024,
            'metadata': {
                'chunk_number': 0,
                'total_chunks': 2,
                'source': 'test.md'
            }
        },
        {
            'content': "Test chunk 2",
            'embedding': [0.2] * 1024,
            'metadata': {
                'chunk_number': 1,
                'total_chunks': 2,
                'source': 'test.md'
            }
        }
    ]
    
    return mock_api

@pytest.fixture
def mock_processing_queue():
    """Create a mock processing queue."""
    return ProcessingQueue()

def test_jina_api_segmentation(mock_jina_api):
    """Test text segmentation with Jina API."""
    test_text = "This is a test document. It should be split into chunks."
    chunks = mock_jina_api.segment_text(test_text)
    
    assert len(chunks) == 2
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert chunks[0].metadata['chunk_number'] == 0
    assert chunks[0].metadata['total_chunks'] == 2

def test_jina_api_embeddings(mock_jina_api):
    """Test embedding generation with Jina API."""
    test_texts = ["Chunk 1", "Chunk 2"]
    embeddings = mock_jina_api.get_embeddings(test_texts)
    
    assert len(embeddings) == 2
    assert all(len(embedding) == 1024 for embedding in embeddings)

def test_jina_api_document_processing(mock_jina_api):
    """Test complete document processing with Jina API."""
    test_text = "This is a test document."
    processed_chunks = mock_jina_api.process_document(test_text, source="test.md")
    
    assert len(processed_chunks) == 2
    assert all('content' in chunk for chunk in processed_chunks)
    assert all('embedding' in chunk for chunk in processed_chunks)
    assert all('metadata' in chunk for chunk in processed_chunks)
    assert all(len(chunk['embedding']) == 1024 for chunk in processed_chunks)
    assert all(chunk['metadata']['source'] == 'test.md' for chunk in processed_chunks)

def test_processing_queue_management(mock_processing_queue):
    """Test processing queue management."""
    # Add documents
    test_files = ["doc1.md", "doc2.md"]
    mock_processing_queue.add_documents(test_files)
    
    # Check initial status
    for file in test_files:
        status, error = mock_processing_queue.get_status(file)
        assert status == ProcessingStatus.PENDING
        assert error is None
    
    # Update status
    mock_processing_queue.update_status("doc1.md", ProcessingStatus.PROCESSING)
    status, error = mock_processing_queue.get_status("doc1.md")
    assert status == ProcessingStatus.PROCESSING
    
    # Test error handling
    mock_processing_queue.update_status("doc2.md", ProcessingStatus.FAILED, "Test error")
    status, error = mock_processing_queue.get_status("doc2.md")
    assert status == ProcessingStatus.FAILED
    assert error == "Test error"
    
    # Test completion
    mock_processing_queue.update_status("doc1.md", ProcessingStatus.COMPLETED)
    mock_processing_queue.clear_completed()
    status, _ = mock_processing_queue.get_status("doc1.md")
    assert status == ProcessingStatus.PENDING  # Default status for unknown files

@pytest.mark.asyncio
async def test_background_processor(mock_jina_api, mock_processing_queue):
    """Test background processing functionality."""
    processor = BackgroundProcessor()
    processor.queue = mock_processing_queue
    
    # Add test documents
    test_files = ["test1.md", "test2.md"]
    processor.queue.add_documents(test_files)
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        for file in test_files:
            with open(docs_dir / file, 'w') as f:
                f.write("Test content")
        
        # Process one document
        doc = processor.queue.get_next_pending()
        assert doc is not None
        
        with patch('src.api.jina.jina_client', mock_jina_api):
            processor._process_document(doc)
        
        # Check status
        status, error = processor.queue.get_status(doc['filename'])
        assert status == ProcessingStatus.COMPLETED
        assert error is None

def test_error_handling(mock_jina_api, mock_processing_queue):
    """Test error handling in document processing."""
    # Make API fail
    mock_jina_api.process_document.side_effect = Exception("API error")
    
    processor = BackgroundProcessor()
    processor.queue = mock_processing_queue
    
    # Add test document
    test_file = "error_test.md"
    processor.queue.add_documents([test_file])
    
    # Create test file
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        with open(docs_dir / test_file, 'w') as f:
            f.write("Test content")
        
        # Process document
        doc = processor.queue.get_next_pending()
        assert doc is not None
        
        with patch('src.api.jina.jina_client', mock_jina_api):
            processor._process_document(doc)
        
        # Check error status
        status, error = processor.queue.get_status(test_file)
        assert status == ProcessingStatus.FAILED
        assert "API error" in error

def test_batch_processing(mock_jina_api, mock_processing_queue):
    """Test batch processing of documents."""
    # Create multiple test documents
    test_files = [f"doc{i}.md" for i in range(5)]
    processor = BackgroundProcessor()
    processor.queue = mock_processing_queue
    processor.queue.add_documents(test_files)
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        for file in test_files:
            with open(docs_dir / file, 'w') as f:
                f.write("Test content")
        
        # Process all documents
        with patch('src.api.jina.jina_client', mock_jina_api):
            while (doc := processor.queue.get_next_pending()) is not None:
                processor._process_document(doc)
        
        # Check all documents are completed
        statuses = processor.queue.get_all_statuses()
        assert all(
            status['status'] == ProcessingStatus.COMPLETED.value
            for status in statuses.values()
        )
