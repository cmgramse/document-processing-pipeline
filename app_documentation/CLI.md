# CLI Manual

## Overview

This document describes all available command-line interface (CLI) commands for the document processing pipeline.

## Environment Setup

Before running any commands, ensure your environment variables are set:
```bash
export JINA_API_KEY=your_jina_api_key
export QDRANT_API_KEY=your_qdrant_api_key
export QDRANT_URL=your_qdrant_url
export QDRANT_COLLECTION_NAME=your_collection
```

## Database Commands

### Initialize Database
```bash
python scripts/init_db.py
```
Creates the SQLite database with proper schema.

### Clean Database
```bash
python scripts/cleanup.py
```
Options:
- `--retention-days DAYS`: Number of days to retain data (default: 30)
- `--force`: Skip confirmation prompts
- `--dry-run`: Show what would be deleted without actually deleting

## Document Processing

### Process Documents
```bash
python -m src.main process [OPTIONS] PATH
```
Process documents in the specified path.

Options:
- `--recursive`: Process documents in subdirectories
- `--force`: Reprocess existing documents
- `--batch-size SIZE`: Number of documents per batch (default: 50)
- `--max-retries N`: Maximum retry attempts (default: 3)

### Validate Documents
```bash
python -m src.main validate PATH
```
Validate documents without processing them.

Options:
- `--show-errors`: Display detailed error information
- `--summary`: Show only summary statistics

## Maintenance Commands

### Cleanup Vectors
```bash
python -m src.main cleanup-vectors
```
Clean up orphaned vectors in Qdrant.

Options:
- `--dry-run`: Show what would be deleted without actually deleting
- `--force`: Skip confirmation prompts

### Optimize Database
```bash
python -m src.main optimize
```
Optimize database performance.

Options:
- `--vacuum`: Run VACUUM command
- `--reindex`: Rebuild indexes
- `--analyze`: Update statistics

## Monitoring Commands

### Show Status
```bash
python -m src.main status
```
Show current processing status.

Options:
- `--watch`: Continuously update status
- `--format {text,json}`: Output format

### Show Statistics
```bash
python -m src.main stats
```
Show processing statistics.

Options:
- `--period {hour,day,week,month}`: Time period
- `--format {text,json,csv}`: Output format

## Queue Management

### List Queue
```bash
python -m src.main queue list
```
List items in processing queue.

Options:
- `--status {pending,processing,failed}`: Filter by status
- `--limit N`: Maximum items to show

### Clear Queue
```bash
python -m src.main queue clear
```
Clear the processing queue.

Options:
- `--status {all,failed,completed}`: Which items to clear
- `--force`: Skip confirmation

## Examples

1. Process all markdown files in docs directory:
```bash
python -m src.main process docs/ --recursive
```

2. Cleanup old data and vectors:
```bash
python scripts/cleanup.py --retention-days 14 --force
```

3. Show processing status with updates:
```bash
python -m src.main status --watch
```

4. Process specific files with custom batch size:
```bash
python -m src.main process docs/file1.md docs/file2.md --batch-size 10
```

5. Validate documents and show errors:
```bash
python -m src.main validate docs/ --show-errors
```

## Error Handling

Common error codes and their meaning:

- `DB001`: Database connection error
- `API001`: Jina AI API error
- `VEC001`: Qdrant vector store error
- `PROC001`: Processing error
- `Q001`: Queue management error

For detailed error information, check the logs in `logs/` directory. 