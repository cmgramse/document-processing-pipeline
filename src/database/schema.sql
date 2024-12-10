-- Documents table: Stores original document information
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    chunk_id INTEGER,
    content TEXT,
    embedding TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    chunking_status TEXT DEFAULT 'pending',
    embedding_status TEXT DEFAULT 'pending',
    qdrant_status TEXT DEFAULT 'pending',
    error_message TEXT,
    version INTEGER DEFAULT 1
);

-- Chunks table: Stores document chunks with vector information
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    chunk_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    chunking_status TEXT DEFAULT 'pending',
    embedding_status TEXT DEFAULT 'pending',
    qdrant_status TEXT DEFAULT 'pending',
    embedding BLOB,
    qdrant_id TEXT,
    processed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_verified_at DATETIME,
    error_message TEXT,
    version INTEGER DEFAULT 1
);

-- Processed files table: Tracks which files have been processed
CREATE TABLE IF NOT EXISTS processed_files (
    filename TEXT PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER,
    status TEXT DEFAULT 'pending',
    chunking_status TEXT DEFAULT 'pending',
    embedding_status TEXT DEFAULT 'pending',
    qdrant_status TEXT DEFAULT 'pending',
    last_verified_at DATETIME,
    error_message TEXT
);

-- Document references table: Tracks relationships between documents
CREATE TABLE IF NOT EXISTS document_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id TEXT NOT NULL,
    target_doc_id TEXT NOT NULL,
    reference_type TEXT NOT NULL,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Processing queue table: Manages background processing tasks
CREATE TABLE IF NOT EXISTS processing_queue (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    task_type TEXT,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Processing history table: Tracks processing events
CREATE TABLE IF NOT EXISTS processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    chunk_id TEXT,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_chunking ON documents(chunking_status);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents(embedding_status);
CREATE INDEX IF NOT EXISTS idx_documents_qdrant ON documents(qdrant_status);
CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename);
CREATE INDEX IF NOT EXISTS idx_chunks_chunking ON chunks(chunking_status);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_status);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant ON chunks(qdrant_status);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id ON chunks(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_processed_files_status ON processed_files(status);
CREATE INDEX IF NOT EXISTS idx_queue_status ON processing_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_document ON processing_queue(document_id);
CREATE INDEX IF NOT EXISTS idx_history_document ON processing_history(document_id);
CREATE INDEX IF NOT EXISTS idx_history_chunk ON processing_history(chunk_id); 