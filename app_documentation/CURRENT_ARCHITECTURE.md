# Current Architecture

## Overview

This document outlines the current architecture of the document processing pipeline. The system processes documents through multiple stages: chunking, embedding generation, and Qdrant vector storage.

## Application Workflow

### 1. Initialization Phase
```
python main.py process
```

1. Database Verification:
   - Check for existing database in `DB_PATH`
   - Initialize new database if none exists
   - Verify schema and recover if needed

2. Document Discovery:
   - List all documents in `DOCUMENTS_PATH`
   - Show supported file types (`.md`, `.txt`)
   - Display relative paths for clarity

3. Interactive Document Selection:
   ```
   Available documents:
   1. guide1.md
   2. guide2.md
   3. notes.txt
   
   Enter document numbers (e.g., "1,3"), ranges (e.g., "1-3"), or "all":
   ```
   - Supports individual selections
   - Supports ranges
   - Supports "all" option
   - Confirms selection before proceeding

4. Duplication Check:
   - Checks if selected documents exist in database
   - If duplicates found:
     ```
     Document 'guide1.md' already exists.
     Options:
     1. Update (removes existing vectors)
     2. Skip
     Select option (1/2):
     ```
   - If update chosen:
     - Removes existing vectors from Qdrant
     - Deletes document info from database
     - Proceeds with upload

### 2. Processing Phase

1. Document Chunking:
   - Uses Jina Segmenter API
   - Creates optimal chunks
   - Updates database status

2. Embedding Generation:
   - Uses Jina Embedding API
   - Processes chunks in batches
   - Updates database status

3. Vector Storage:
   - Uploads to Qdrant
   - Verifies successful upload
   - Updates database status

4. Status Tracking:
   - Real-time progress updates
   - Error reporting
   - Final statistics

### 3. Summary Phase

```
Processing Complete:

Documents:
- Processed: 3
- Failed: 0
- Updated: 1

Chunks:
- Total: 15
- Successfully embedded: 15
- Successfully uploaded: 15

Processing time: 45.2 seconds
Average chunk processing time: 3.01 seconds
```

## Components

### 1. Database Layer (SQLAlchemy ORM)

The database layer uses SQLAlchemy ORM for robust data management:

#### Models (`models.py`)
- `Document`: Represents source documents
- `Chunk`: Represents document segments
- `ProcessedFile`: Tracks file processing status
- `ProcessingHistory`: Logs processing actions

#### Session Management (`session.py`)
- Connection pooling
- Transaction management
- WAL mode support
- Safe session handling

#### Operations (`operations.py`)
- High-level database operations
- Status management
- Processing tracking
- Statistics gathering

### 2. Processing Layer

#### Document Processing (`documents.py`)
- Document chunking
- Token counting
- Status tracking
- Error handling

#### Embedding Generation (`embeddings.py`)
- Jina AI API integration
- Embedding generation
- Batch processing
- Error recovery

#### Vector Storage (`qdrant.py`)
- Qdrant API integration
- Vector upsert operations
- Collection management
- Status synchronization

### 3. Main Processing Pipeline

The pipeline follows these steps:

1. Document Chunking:
   - Split documents into manageable chunks
   - Track chunks in database using SQLAlchemy
   - Calculate token counts

2. Embedding Generation:
   - Generate embeddings via Jina AI
   - Store embeddings in database
   - Track embedding status

3. Vector Storage:
   - Upload vectors to Qdrant
   - Track upload status
   - Verify vector presence

## State Management

### Status Fields
Each chunk has multiple status fields:
- `chunking_status`
- `embedding_status`
- `qdrant_status`

Valid states:
- `pending`
- `processing`
- `completed`
- `failed`
- `retrying`

### State Validation
- Model-level validation via SQLAlchemy
- Status transition rules
- Relationship integrity checks

## Error Handling

1. Database Errors:
   - Automatic transaction rollback
   - Session cleanup
   - Error logging

2. API Errors:
   - Retry mechanisms
   - Error tracking
   - Status updates

3. Processing Errors:
   - Safe state transitions
   - Error message capture
   - Recovery procedures

## Monitoring

1. Processing Statistics:
   - Document counts
   - Chunk status
   - Error rates
   - Processing times

2. System Health:
   - Database connections
   - API availability
   - Resource usage

## Security

1. Database:
   - SQLAlchemy validation
   - Safe session handling
   - Transaction isolation

2. API Keys:
   - Environment variables
   - Secure storage
   - Access control

## Future Improvements

1. Performance:
   - Query optimization
   - Batch processing
   - Caching strategies

2. Monitoring:
   - Real-time metrics
   - Performance dashboards
   - Alert system

3. Features:
   - Async processing
   - Bulk operations
   - Migration tools
  