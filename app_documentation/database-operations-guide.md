# Database Operations Guide

## Overview

This document explains the organization and purpose of the database operations layer, which now uses SQLAlchemy ORM for better data management and safety.

## Automatic Database Initialization

The system automatically handles database initialization and verification. This process happens automatically whenever you run any command:

1. Database Creation:
```python
# Happens automatically when needed
python main.py process  # or any other command
```

The system will:
- Check if database exists at the configured location
- Create parent directories if needed
- Initialize the database if it doesn't exist
- Verify schema integrity
- Attempt recovery if issues are found

### Initialization Process

1. Location Check:
   - Uses `DB_PATH` from environment (default: `data/documents.db`)
   - Creates parent directories if missing
   - Checks if database file exists

2. Schema Verification:
   - Tests database connection
   - Verifies required tables exist (`document`, `chunk`)
   - Checks schema completeness

3. Error Recovery:
   - Attempts to fix corrupted databases
   - Reinitializes if schema is incomplete
   - Provides clear error messages

### Status Messages

You'll see different messages depending on the database state:

1. First Run:
```
[INFO] Initializing new database...
[INFO] Database initialized at data/documents.db
[INFO] Database connection verified
```

2. Normal Operation:
```
[INFO] Found N documents to process...
```

3. Recovery:
```
[WARNING] Database schema incomplete, reinitializing...
[INFO] Database schema restored
```

### Error Handling

The system handles various error conditions:

1. Missing Database:
   - Creates new database automatically
   - Initializes required schema
   - Verifies successful creation

2. Corrupted Schema:
   - Detects missing or incomplete tables
   - Attempts automatic recovery
   - Reinitializes if necessary

3. Connection Issues:
   - Validates database connection
   - Reports specific error messages
   - Attempts recovery when possible

## Configuration

### Environment Variables

The application uses the following environment variables:

1. Database Configuration:
```bash
# Database location
DB_PATH=data/documents.db

# Connection pool settings
DB_POOL_SIZE=5          # Number of connections to maintain
DB_MAX_OVERFLOW=10      # Maximum extra connections when pool is full
DB_POOL_TIMEOUT=30      # Seconds to wait for available connection
DB_POOL_RECYCLE=1800    # Seconds before recycling a connection
```

2. Jina AI Configuration:
```bash
# API credentials and model selection
JINA_API_KEY=your-jina-api-key-here
JINA_EMBEDDING_MODEL=jina-embeddings-v3
```

3. Qdrant Configuration:
```bash
# Vector database settings
QDRANT_API_KEY=your-qdrant-api-key-here
QDRANT_URL=your-qdrant-url-here
QDRANT_COLLECTION_NAME=your-collection-name
```

4. Document Processing:
```bash
# Input documents location
DOCUMENTS_PATH=docs

# Logging level
LOG_LEVEL=INFO
```

### Database Path

The database location can be configured using the `DB_PATH` environment variable:

```bash
# In .env file or environment
DB_PATH=data/documents.db  # Default value
```

You can customize the database location by:
1. Setting `DB_PATH` in your `.env` file
2. Setting it as an environment variable
3. Using the default value (`data/documents.db`)

The system will:
- Create parent directories if they don't exist
- Use the specified path for all database operations
- Show the actual path in logs and status messages

### Connection Pool

SQLAlchemy uses a connection pool for better performance:

1. Pool Size:
   - `DB_POOL_SIZE`: Number of connections to maintain
   - `DB_MAX_OVERFLOW`: Extra connections allowed when pool is full

2. Connection Lifecycle:
   - `DB_POOL_TIMEOUT`: Seconds to wait for available connection
   - `DB_POOL_RECYCLE`: Seconds before recycling a connection

## Database Operations

### Basic Operations

1. Document Processing:
```bash
python main.py process
```
- Processes new documents
- Updates existing records
- Handles chunking and embeddings

2. Status Check:
```bash
python main.py stats
```
- Shows processing statistics
- Displays system metrics
- Reports database status

3. Cleanup:
```bash
python main.py cleanup
```
- Cleans up database
- Synchronizes with Qdrant
- Removes orphaned records

### Document Management

1. Interactive Deletion:
```bash
python main.py delete
```
- Lists available documents
- Allows selective deletion
- Confirms before removing

2. Specific Document Deletion:
```bash
python main.py delete doc1.txt doc2.txt
```
- Deletes specified documents
- Removes associated chunks
- Updates related records

## Core Components

### 1. Models (`models.py`)

SQLAlchemy models that define our database schema and relationships:

```python
class Document(Base):
    __tablename__ = 'documents'
    # ... columns and relationships
    
class Chunk(Base):
    __tablename__ = 'chunks'
    # ... columns and relationships
```

Key features:
- Automatic schema validation
- Relationship management
- State validation via `@validates` decorators
- Type safety and constraints

### 2. Session Management (`session.py`)

Handles database connections and transaction management:

```python
from sqlalchemy.orm import sessionmaker

# Create session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

@contextmanager
def get_db() -> Session:
    """Get a database session context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

Features:
- Connection pooling
- Automatic transaction management
- Safe session handling
- WAL mode support

### 3. Operations (`operations.py`)

High-level database operations using SQLAlchemy ORM:

```python
def mark_file_as_processed(session: Session, filename: str, chunk_count: int):
    """Mark a file as processed using SQLAlchemy."""
    document = session.query(Document).filter_by(filename=filename).first()
    # ... update document status
    session.commit()
```

Key functions:
- `mark_file_as_processed`: Update file processing status
- `track_document_chunk`: Create new chunk records
- `get_pending_chunks`: Retrieve chunks for processing
- `update_chunk_status`: Update chunk processing state

## Best Practices

1. Session Management:
   ```python
   with get_db() as session:
       # Your database operations here
       document = session.query(Document).filter_by(id=doc_id).one()
       document.status = 'completed'
       # Session will automatically commit or rollback
   ```

2. State Validation:
   ```python
   @validates('status')
   def validate_status(self, key, status):
       valid_statuses = {'pending', 'processing', 'completed', 'failed'}
       if status not in valid_statuses:
           raise ValueError(f"Invalid status: {status}")
       return status
   ```

3. Relationship Usage:
   ```python
   # Access related chunks through relationship
   document = session.query(Document).first()
   for chunk in document.chunks:
       print(chunk.status)
   ```

## Error Handling

1. Session Errors:
   - Automatic rollback on exception
   - Transaction safety
   - Proper resource cleanup

2. Validation Errors:
   - Model-level validation
   - Relationship integrity
   - State transition rules

## Monitoring

1. Status Tracking:
   ```python
   def get_processing_stats(session: Session) -> Dict[str, int]:
       """Get current processing statistics."""
       return {
           'total_chunks': session.query(Chunk).count(),
           'pending_chunks': session.query(Chunk).filter_by(status='pending').count(),
           # ... more stats
       }
   ```

2. Performance Monitoring:
   - Query execution tracking
   - Connection pool stats
   - Transaction duration

## Migration from Raw SQL

The codebase has been migrated from raw SQL to SQLAlchemy ORM for:
- Better type safety
- Automatic schema management
- Improved transaction handling
- Better relationship management
- More maintainable code

Key changes:
1. Replaced raw SQL queries with ORM queries
2. Added model validation
3. Improved transaction safety
4. Added relationship management
5. Better error handling

## Future Considerations

1. Performance Optimization:
   - Query optimization
   - Indexing strategies
   - Caching implementation

2. Additional Features:
   - Bulk operations
   - Async support
   - Migration tools 