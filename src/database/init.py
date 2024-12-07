import sqlite3
import logging
from pathlib import Path

def initialize_database():
    """Initialize SQLite database with proper schema and indices"""
    logging.info("Setting up SQLite database...")
    
    # Create docs directory if it doesn't exist
    docs_path = Path('./docs')
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logging.info("Created docs directory")
    
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON")
    
    # Create chunks table to store segmenter results
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  chunk_number INTEGER,
                  content TEXT,
                  token_count INTEGER,
                  embedding_status TEXT DEFAULT 'pending',  -- 'pending', 'completed', 'failed'
                  created_at TIMESTAMP,
                  processed_at TIMESTAMP,
                  UNIQUE(filename, chunk_number))''')
    
    # Create documents table for final processed chunks with embeddings
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id TEXT PRIMARY KEY,
                  chunk_id TEXT,  -- References chunks.id
                  filename TEXT,
                  content TEXT,
                  embedding_id TEXT,
                  embedding BLOB,
                  qdrant_status TEXT DEFAULT 'pending',  -- 'pending', 'uploaded', 'failed'
                  processed_at TIMESTAMP,
                  uploaded_at TIMESTAMP,
                  FOREIGN KEY(chunk_id) REFERENCES chunks(id))''')
    
    # Create processed_files table
    c.execute('''CREATE TABLE IF NOT EXISTS processed_files
                 (filename TEXT PRIMARY KEY,
                  last_modified FLOAT,
                  processed_at TIMESTAMP,
                  chunk_count INTEGER,
                  status TEXT DEFAULT 'segmented')''')  -- 'segmented', 'embedded', 'uploaded'
    
    # Create indices for better query performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(embedding_status)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_docs_filename ON documents(filename)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_qdrant_status ON documents(qdrant_status)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_processed_status ON processed_files(status)')
    
    conn.commit()
    return conn