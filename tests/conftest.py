"""Test configuration and fixtures."""
import os
import tempfile
import sqlite3
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from datetime import datetime

@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    os.environ['QDRANT_COLLECTION_NAME'] = 'test_collection'
    yield
    if 'QDRANT_COLLECTION_NAME' in os.environ:
        del os.environ['QDRANT_COLLECTION_NAME']

@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    _, db_path = tempfile.mkstemp()
    conn = sqlite3.connect(db_path)
    
    # Create test schema
    c = conn.cursor()
    
    # Create chunks table
    c.execute('''CREATE TABLE chunks
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  chunk_number INTEGER,
                  content TEXT,
                  token_count INTEGER,
                  embedding_status TEXT DEFAULT 'pending',
                  created_at TIMESTAMP,
                  processed_at TIMESTAMP,
                  content_hash TEXT,
                  version INTEGER DEFAULT 1,
                  document_id TEXT,
                  embedding TEXT,
                  embedding_model TEXT,
                  UNIQUE(filename, chunk_number))''')
    
    # Create documents table
    c.execute('''CREATE TABLE documents
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  chunk_id TEXT,
                  content TEXT,
                  content_hash TEXT,
                  embedding TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  status TEXT DEFAULT 'pending',
                  last_modified TIMESTAMP,
                  version_status TEXT DEFAULT 'pending',
                  FOREIGN KEY(chunk_id) REFERENCES chunks(id))''')
    
    # Create processed_files table
    c.execute('''CREATE TABLE processed_files
                 (filename TEXT PRIMARY KEY,
                  last_modified FLOAT,
                  processed_at TIMESTAMP,
                  chunk_count INTEGER,
                  status TEXT DEFAULT 'segmented')''')
    
    # Create document_references table
    c.execute('''CREATE TABLE document_references
                 (id TEXT PRIMARY KEY,
                  source_doc_id TEXT,
                  target_doc_id TEXT,
                  ref_type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(source_doc_id) REFERENCES documents(id),
                  FOREIGN KEY(target_doc_id) REFERENCES documents(id))''')
    
    conn.commit()
    
    yield conn
    
    conn.close()
    os.unlink(db_path)

@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = MagicMock()
    
    # Mock successful upload
    client.upload_documents.return_value = True
    
    # Mock successful search
    client.similarity_search.return_value = [
        MagicMock(page_content="Test content", metadata={"source": "test.md"})
    ]
    
    return client

@pytest.fixture
def test_docs_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        # Create test markdown files
        files = [
            ("test1.md", "# Test Document 1\n\nThis is a test document with multiple paragraphs.\n\nIt has enough content to be split into chunks."),
            ("test2.md", "This is test document 2"),
            ("test3.md", "This is test document 3")
        ]
        
        for filename, content in files:
            with open(docs_dir / filename, 'w') as f:
                f.write(content)
        
        yield docs_dir

@pytest.fixture
def mock_jina_api(monkeypatch):
    """Mock Jina AI API responses."""
    def mock_segment_text(text, api_key=None):
        # For testing, return simple chunks if no API key
        if not api_key:
            text_length = len(text)
            
            # For very short text, return as single chunk
            if text_length < 100:
                return [{"text": text, "start": 0, "end": text_length, "metadata": {}}]
            
            # For medium text, split into 2 chunks at nearest space
            elif text_length < 500:
                mid = text_length // 2
                # Find nearest space to split
                while mid > 0 and text[mid] != ' ':
                    mid -= 1
                if mid == 0:  # No space found, force split
                    mid = text_length // 2
                return [
                    {"text": text[:mid], "start": 0, "end": mid, "metadata": {}},
                    {"text": text[mid:], "start": mid, "end": text_length, "metadata": {}}
                ]
            
            # For long text, split into multiple chunks of roughly equal size
            else:
                chunks = []
                num_chunks = 3
                chunk_size = text_length // num_chunks
                current_pos = 0
                for i in range(num_chunks):
                    end_pos = min(current_pos + chunk_size, text_length)
                    # Find nearest space to split, unless it's the last chunk
                    if end_pos < text_length and i < num_chunks - 1:
                        while end_pos > current_pos and text[end_pos] != ' ':
                            end_pos -= 1
                        if end_pos == current_pos:  # No space found, force split
                            end_pos = min(current_pos + chunk_size, text_length)
                    chunk = {
                        "text": text[current_pos:end_pos],
                        "start": current_pos,
                        "end": end_pos,
                        "metadata": {}
                    }
                    chunks.append(chunk)
                    current_pos = end_pos
                return chunks
    
    def mock_post(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def raise_for_status(self):
                pass
            
            def json(self):
                # Mock the response format from Jina AI API
                input_texts = kwargs.get('json', {}).get('input', [])
                data = []
                for i, text in enumerate(input_texts):
                    data.append({
                        "embedding": [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)],
                        "index": i,
                        "text": text,
                        "embedding_model": "jina-embeddings-v3",
                        "embedding_version": "1.0.0"
                    })
                
                return {
                    "data": data,
                    "model": "jina-embeddings-v3",
                    "usage": {"prompt_tokens": len(input_texts) * 10, "total_tokens": len(input_texts) * 10}
                }
        return MockResponse()
    
    monkeypatch.setenv("JINA_API_KEY", "test_key")
    # Update the mock paths to match the actual imports
    monkeypatch.setattr("src.api.jina.segment_text", mock_segment_text)
    monkeypatch.setattr("requests.post", mock_post)

@pytest.fixture
def sample_processed_doc(temp_db):
    """Create a sample processed document in the database."""
    c = temp_db.cursor()
    
    # Create a processed file entry
    c.execute("""
        INSERT INTO processed_files (filename, last_modified, processed_at, chunk_count, status)
        VALUES (?, ?, ?, ?, ?)
    """, ('test.md', 123.45, datetime.now(), 2, 'embedded'))
    
    # Create chunk entries
    chunk_ids = []
    for i in range(2):
        chunk_id = f'chunk_{i}'
        chunk_ids.append(chunk_id)
        c.execute("""
            INSERT INTO chunks 
            (id, filename, chunk_number, content, token_count, embedding_status, created_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, 'test.md', i, f'content_{i}', 10, 'completed', datetime.now(), f'hash_{i}'))
    
    # Create document entries
    for i, chunk_id in enumerate(chunk_ids):
        c.execute("""
            INSERT INTO documents 
            (id, filename, chunk_id, content, content_hash, embedding, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f'doc_{i}', 'test.md', chunk_id, f'content_{i}', f'hash_{i}', 'embedding', 'uploaded'))
    
    temp_db.commit()
    return 'test.md'
