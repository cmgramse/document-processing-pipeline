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
                  content TEXT,
                  token_count INTEGER,
                  embedding_status TEXT DEFAULT 'pending',
                  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  version INTEGER DEFAULT 1)''')
    
    # Create documents table
    c.execute('''CREATE TABLE documents
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  chunk_id INTEGER,
                  content TEXT,
                  embedding TEXT,
                  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create processed_files table
    c.execute('''CREATE TABLE processed_files
                 (filename TEXT PRIMARY KEY,
                  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  chunk_count INTEGER,
                  status TEXT DEFAULT 'pending')''')
    
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
    """Mock Jina API responses."""
    def mock_segment_text(text, api_key=None, chunk_size=1000, overlap=100):
        # Return chunks based on text length
        text_length = len(text)
        
        # For very short text, return as single chunk
        if text_length < 20:
            return [{"text": text, "start": 0, "end": text_length, "metadata": {}}]
        
        # For medium text, split into 2 chunks at nearest space
        elif text_length < 200:
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

    def mock_get_embeddings(chunks, api_key=None, batch_size=50):
        # Return mock embeddings for each chunk
        return {
            "data": [
                {
                    "text": chunk["text"],
                    "embedding": [0.1, 0.2, 0.3],  # Mock embedding
                    "embedding_model": "test-model",
                    "embedding_version": "v1",
                    "index": i
                }
                for i, chunk in enumerate(chunks)
            ],
            "model": "test-model",
            "usage": {"total_tokens": 100}
        }

    def mock_getenv(key, default=None):
        """Mock environment variables."""
        if key == 'JINA_API_KEY':
            return 'test_key'
        return default

    def mock_post(*args, **kwargs):
        """Mock requests.post."""
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code

            def json(self):
                return self.json_data

            def raise_for_status(self):
                if self.status_code != 200:
                    raise Exception(f"HTTP {self.status_code}")

        # Mock segment_text response
        if args[0] == "https://segment.jina.ai/":
            text = kwargs["json"]["content"]
            text_length = len(text)
            
            # For very short text, return as single chunk
            if text_length < 20:
                chunks = [{"text": text, "start": 0, "end": text_length, "metadata": {}}]
            
            # For medium text, split into 2 chunks at nearest space
            elif text_length < 200:
                mid = text_length // 2
                # Find nearest space to split
                while mid > 0 and text[mid] != ' ':
                    mid -= 1
                if mid == 0:  # No space found, force split
                    mid = text_length // 2
                chunks = [
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
            
            return MockResponse({"chunks": chunks}, 200)

        # Mock get_embeddings response
        elif args[0] == "https://api.jina.ai/v1/embeddings":
            input_texts = kwargs["json"]["input"]
            return MockResponse({
                "data": [
                    {
                        "text": text,
                        "embedding": [0.1, 0.2, 0.3],  # Mock embedding
                        "embedding_model": "test-model",
                        "embedding_version": "v1",
                        "index": i
                    }
                    for i, text in enumerate(input_texts)
                ],
                "model": "test-model",
                "usage": {"total_tokens": 100}
            }, 200)

        return MockResponse({}, 404)

    monkeypatch.setattr("src.api.jina.segment_text", mock_segment_text)
    monkeypatch.setattr("src.api.jina.get_embeddings", mock_get_embeddings)
    monkeypatch.setattr("os.getenv", mock_getenv)
    monkeypatch.setattr("requests.post", mock_post)

@pytest.fixture
def sample_processed_doc(temp_db):
    """Create a sample processed document in the database."""
    c = temp_db.cursor()
    
    # Create a processed file entry
    c.execute("""
        INSERT INTO processed_files (filename, processed_at, chunk_count, status)
        VALUES (?, datetime('now'), ?, ?)
    """, ('test.md', 2, 'embedded'))
    
    # Create chunk entries
    chunk_ids = []
    for i in range(2):
        chunk_id = f'chunk_{i}'
        chunk_ids.append(chunk_id)
        c.execute("""
            INSERT INTO chunks 
            (id, filename, content, token_count, embedding_status)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, 'test.md', f'content_{i}', 10, 'completed'))
    
    # Create document entries
    for i, chunk_id in enumerate(chunk_ids):
        c.execute("""
            INSERT INTO documents 
            (id, filename, chunk_id, content, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, 'test.md', i, f'content_{i}', '[0.1, 0.2, 0.3]'))
    
    temp_db.commit()
    return 'test.md'
