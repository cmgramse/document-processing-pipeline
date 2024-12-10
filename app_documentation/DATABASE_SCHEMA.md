# Database Schema Documentation

## Overview
The database schema is designed to support document processing, chunking, embedding generation, and vector storage operations. It uses SQLite with WAL journaling mode for better concurrency.

## Tables

### 1. documents
Primary table for document tracking.

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,                                    -- Unique document identifier
    filename TEXT NOT NULL,                                 -- Original filename
    chunk_id INTEGER,                                       -- Reference to chunk
    content TEXT,                                          -- Document content
    embedding TEXT,                                        -- Document-level embedding
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,       -- Last processing timestamp
    status TEXT DEFAULT 'pending',                         -- Overall status
    chunking_status TEXT DEFAULT 'pending',                -- Chunking process status
    embedding_status TEXT DEFAULT 'pending',               -- Embedding generation status
    qdrant_status TEXT DEFAULT 'pending',                  -- Qdrant upload status
    error_message TEXT,                                    -- Error details if any
    version INTEGER DEFAULT 1                              -- Schema version
);
```

### 2. chunks
Stores individual document chunks and their processing status.

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,                                    -- Unique chunk identifier
    filename TEXT NOT NULL,                                 -- Source document filename
    content TEXT NOT NULL,                                  -- Chunk content
    token_count INTEGER,                                    -- Number of tokens in chunk
    chunk_number INTEGER NOT NULL,                          -- Sequence number in document
    content_hash TEXT NOT NULL,                             -- Content verification hash
    chunking_status TEXT DEFAULT 'pending',                 -- Chunking process status
    embedding_status TEXT DEFAULT 'pending',                -- Embedding generation status
    qdrant_status TEXT DEFAULT 'pending',                   -- Qdrant upload status
    embedding BLOB,                                         -- Vector embedding data
    qdrant_id TEXT,                                         -- Qdrant point identifier
    processed_at DATETIME,                                  -- Last processing time
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,          -- Creation timestamp
    last_verified_at DATETIME,                              -- Last verification time
    error_message TEXT,                                     -- Error details if any
    version INTEGER DEFAULT 1                               -- Schema version
);
```

### 3. processed_files
Tracks overall file processing status and history.

```sql
CREATE TABLE processed_files (
    filename TEXT PRIMARY KEY,                              -- Unique filename
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,       -- Processing timestamp
    chunk_count INTEGER,                                    -- Total chunks created
    status TEXT DEFAULT 'pending',                          -- Overall status
    chunking_status TEXT DEFAULT 'pending',                 -- Chunking process status
    embedding_status TEXT DEFAULT 'pending',                -- Embedding generation status
    qdrant_status TEXT DEFAULT 'pending',                   -- Qdrant upload status
    last_verified_at DATETIME,                              -- Last verification time
    error_message TEXT                                      -- Error details if any
);
```

### 4. document_references
Manages relationships between documents.

```sql
CREATE TABLE document_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,                   -- Unique reference ID
    source_doc_id TEXT NOT NULL,                           -- Source document ID
    target_doc_id TEXT NOT NULL,                           -- Target document ID
    reference_type TEXT NOT NULL,                          -- Type of reference
    context TEXT,                                          -- Reference context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,         -- Creation timestamp
    FOREIGN KEY (source_doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

### 5. processing_queue
Manages document processing queue.

```sql
CREATE TABLE processing_queue (
    id TEXT PRIMARY KEY,                                    -- Unique queue entry ID
    document_id TEXT,                                       -- Document to process
    task_type TEXT,                                         -- Type of processing task
    status TEXT DEFAULT 'pending',                          -- Queue status
    priority INTEGER DEFAULT 0,                             -- Processing priority
    error_message TEXT,                                     -- Error details if any
    retry_count INTEGER DEFAULT 0,                          -- Number of retries
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,         -- Creation timestamp
    updated_at TIMESTAMP,                                   -- Last update time
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

### 6. processing_history
Tracks processing history and actions.

```sql
CREATE TABLE processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,                   -- Unique history ID
    document_id TEXT,                                       -- Related document
    chunk_id TEXT,                                          -- Related chunk
    action TEXT NOT NULL,                                   -- Action performed
    status TEXT NOT NULL,                                   -- Action status
    details JSON,                                           -- Action details
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,         -- Creation timestamp
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);
```

## Indexes

### Document Indexes
- `idx_documents_filename`: Optimize document lookup by filename
- `idx_documents_status`: Filter documents by status
- `idx_documents_chunking`: Filter by chunking status
- `idx_documents_embedding`: Filter by embedding status
- `idx_documents_qdrant`: Filter by Qdrant status

### Chunk Indexes
- `idx_chunks_filename`: Group chunks by source document
- `idx_chunks_chunking`: Filter by chunking status
- `idx_chunks_embedding`: Filter by embedding status
- `idx_chunks_qdrant`: Filter by Qdrant status
- `idx_chunks_qdrant_id`: Quick Qdrant ID lookup

### Queue and History Indexes
- `idx_processed_files_status`: Filter processed files by status
- `idx_queue_status`: Filter queue by status
- `idx_queue_document`: Group queue entries by document
- `idx_history_document`: Group history by document
- `idx_history_chunk`: Group history by chunk

## Status Values
Common status values across tables:
- `pending`: Initial state
- `processing`: Currently being processed
- `completed`: Successfully processed
- `failed`: Processing failed
- `retrying`: Failed but will retry 