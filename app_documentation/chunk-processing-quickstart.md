# Chunk Processing Quick Reference

## Overview

This guide provides quick reference for common chunk processing operations using the transaction management system.

## Common Operations

### 1. Process Single Chunk

```python
from database.transaction import chunk_transaction, update_chunk_status

def process_single_chunk(conn, chunk_id):
    try:
        # Get chunk data
        with chunk_transaction(conn, chunk_id) as cursor:
            cursor.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,))
            content = cursor.fetchone()[0]
            
        # Generate embedding
        embedding = generate_embedding(content)
        
        # Upload to Qdrant
        qdrant_id = upload_to_qdrant(embedding)
        
        # Update status
        update_chunk_status(
            conn=conn,
            chunk_id=chunk_id,
            status='completed',
            qdrant_id=qdrant_id,
            embedding=embedding
        )
        
    except Exception as e:
        logger.error(f"Failed to process chunk {chunk_id}: {e}")
        update_chunk_status(conn, chunk_id, 'failed')
```

### 2. Batch Processing

```python
def process_chunk_batch(conn, chunk_ids):
    results = {'success': [], 'failed': []}
    
    for chunk_id in chunk_ids:
        try:
            with chunk_transaction(conn, chunk_id) as cursor:
                # Your processing logic here
                results['success'].append(chunk_id)
        except Exception as e:
            results['failed'].append((chunk_id, str(e)))
            
    return results
```

### 3. Check Chunk Status

```python
def get_chunk_status(conn, chunk_id):
    with chunk_transaction(conn, chunk_id) as cursor:
        cursor.execute("""
            SELECT embedding_status, qdrant_status, qdrant_id, error_message
            FROM chunks WHERE id = ?
        """, (chunk_id,))
        return cursor.fetchone()
```

### 4. Retry Failed Chunks

```python
def retry_failed_chunks(conn, max_retries=3):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM chunks 
        WHERE status = 'failed' 
        AND retry_count < ?
    """, (max_retries,))
    
    failed_chunks = [row[0] for row in cursor.fetchall()]
    return process_chunk_batch(conn, failed_chunks)
```

## Common Patterns

### 1. State Transitions

```python
# Valid state transitions
VALID_TRANSITIONS = {
    'pending': ['processing', 'failed'],
    'processing': ['completed', 'failed'],
    'completed': ['verified', 'failed'],
    'failed': ['pending']  # Only for retries
}
```

### 2. Error Handling

```python
def safe_chunk_operation(conn, chunk_id, operation):
    try:
        with chunk_transaction(conn, chunk_id) as cursor:
            result = operation(cursor)
            return result
    except Exception as e:
        logger.error(f"Operation failed for chunk {chunk_id}: {e}")
        update_chunk_status(conn, chunk_id, 'failed')
        raise
```

### 3. Batch Operations

```python
def chunked_processing(conn, chunk_ids, batch_size=50):
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i:i + batch_size]
        process_chunk_batch(conn, batch)
```

## Common Issues and Solutions

### 1. Lost Updates

**Problem:**
```python
# DON'T DO THIS
conn.execute("UPDATE chunks SET status = ? WHERE id = ?", ('completed', chunk_id))
```

**Solution:**
```python
# DO THIS
with chunk_transaction(conn, chunk_id) as cursor:
    cursor.execute("UPDATE chunks SET status = ? WHERE id = ?", ('completed', chunk_id))
```

### 2. State Inconsistency

**Problem:**
```python
# DON'T DO THIS
cursor.execute("UPDATE chunks SET qdrant_status = 'completed' WHERE id = ?")
```

**Solution:**
```python
# DO THIS
update_chunk_status(conn, chunk_id, 'completed', qdrant_id=qdrant_id)
```

### 3. Connection Issues

**Problem:**
```python
# DON'T DO THIS
conn = sqlite3.connect('database.db')
# Use connection in multiple threads
```

**Solution:**
```python
# DO THIS
def worker():
    with get_connection() as conn:
        # Use connection in single thread
```

## Best Practices Checklist

- [ ] Always use `chunk_transaction` for database operations
- [ ] Verify state consistency after updates
- [ ] Handle errors and rollback transactions
- [ ] Use proper connection management
- [ ] Log operations and state transitions
- [ ] Implement retry mechanisms for failed operations
- [ ] Use batch processing for better performance
- [ ] Monitor transaction duration and success rates

## Quick Troubleshooting

1. **Transaction fails to commit:**
   - Check for proper cursor closure
   - Verify transaction boundaries
   - Look for nested transactions

2. **State verification fails:**
   - Check state transition logic
   - Verify atomic updates
   - Look for concurrent modifications

3. **Performance issues:**
   - Use appropriate batch sizes
   - Check index usage
   - Monitor transaction duration

## Useful SQL Queries

### Check Chunk Status
```sql
SELECT 
    COUNT(*) as count,
    embedding_status,
    qdrant_status
FROM chunks
GROUP BY embedding_status, qdrant_status;
```

### Find Stuck Chunks
```sql
SELECT id, status, error_message
FROM chunks
WHERE processed_at < datetime('now', '-1 hour')
AND status IN ('processing', 'pending');
```

### Check Error Distribution
```sql
SELECT error_message, COUNT(*) as count
FROM chunks
WHERE status = 'failed'
GROUP BY error_message
ORDER BY count DESC;