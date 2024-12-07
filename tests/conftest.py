"""Test configuration and fixtures."""
import os
import tempfile
import sqlite3
import pytest
from unittest.mock import MagicMock
from pathlib import Path

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
                  UNIQUE(filename, chunk_number))''')
    
    # Create documents table
    c.execute('''CREATE TABLE documents
                 (id TEXT PRIMARY KEY,
                  chunk_id TEXT,
                  filename TEXT,
                  content TEXT,
                  embedding_id TEXT,
                  embedding BLOB,
                  qdrant_status TEXT DEFAULT 'pending',
                  processed_at TIMESTAMP,
                  uploaded_at TIMESTAMP,
                  FOREIGN KEY(chunk_id) REFERENCES chunks(id))''')
    
    # Create processed_files table
    c.execute('''CREATE TABLE processed_files
                 (filename TEXT PRIMARY KEY,
                  last_modified FLOAT,
                  processed_at TIMESTAMP,
                  chunk_count INTEGER,
                  status TEXT DEFAULT 'segmented')''')
    
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
            ("test1.md", "This is test document 1"),
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
    def mock_segment_text(*args, **kwargs):
        return {
            "chunks": ["Chunk 1", "Chunk 2"],
            "num_tokens": 20
        }
    
    def mock_get_embeddings(*args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in range(len(args[0]))]
    
    monkeypatch.setenv("JINA_API_KEY", "test_key")
    monkeypatch.setattr("src.api.jina.segment_text", mock_segment_text)
    monkeypatch.setattr("src.api.jina.get_embeddings", mock_get_embeddings)

@pytest.fixture
def sample_processed_doc(temp_db):
    """Create a sample processed document in the database."""
    c = temp_db.cursor()
    
    # Add a processed file
    c.execute("""
        INSERT INTO processed_files (filename, last_modified, processed_at, chunk_count, status)
        VALUES (?, ?, datetime('now'), ?, ?)
    """, ("test.md", 123.45, 2, "embedded"))
    
    # Add chunks
    for i in range(2):
        c.execute("""
            INSERT INTO chunks (id, filename, chunk_number, content, token_count, 
                              embedding_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (f"chunk_{i}", "test.md", i, f"Content {i}", 10, "completed"))
    
    # Add documents
    for i in range(2):
        c.execute("""
            INSERT INTO documents (id, chunk_id, filename, content, embedding_id, 
                                 embedding, qdrant_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f"doc_{i}", f"chunk_{i}", "test.md", f"Content {i}", 
              f"emb_{i}", "[0.1, 0.2, 0.3]", "uploaded"))
    
    temp_db.commit()
    return "test.md"
