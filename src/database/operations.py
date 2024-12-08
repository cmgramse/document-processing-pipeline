"""
Database Operations Module

This module handles all SQLite database operations for the document management system.
It provides functions for document tracking, status updates, and statistics gathering.

The module manages:
- Document processing status
- Chunk metadata and versions
- Processing statistics
- Document-chunk relationships

Functions in this module are designed to be atomic and transactional to ensure
database consistency even during concurrent operations.

Example:
    Track a new document:
        doc_id = track_document('example.md', conn)
    
    Update processing status:
        update_processing_status(doc_id, 'completed', conn)
    
    Get document statistics:
        stats = get_document_stats(doc_id, conn)
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import hashlib
from pathlib import Path
import os

def get_unprocessed_files(conn: sqlite3.Connection, available_docs: List[str]) -> List[str]:
    """
    Filter out already processed files that haven't been modified since

    Args:
        conn: SQLite database connection
        available_docs: List of available document paths

    Returns:
        List of unprocessed file paths
    """
    c = conn.cursor()
    
    # Get list of processed files
    c.execute("SELECT filename FROM processed_files WHERE status = 'embedded'")
    processed = {os.path.basename(row[0]) for row in c.fetchall()}
    
    # Return files that haven't been processed
    return [doc for doc in available_docs if os.path.basename(doc) not in processed]

def mark_file_as_processed(conn: sqlite3.Connection, filename: str, chunk_count: int) -> None:
    """
    Mark a file as processed in the database.
    
    Args:
        conn: SQLite database connection
        filename: Path of the file to mark as processed
        chunk_count: Number of chunks created from the file
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-mark-processed".encode()).hexdigest()[:8]
    
    try:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO processed_files 
            (filename, processed_at, chunk_count, status)
            VALUES (?, datetime('now'), ?, 'embedded')
        """, (os.path.basename(filename), chunk_count))
        conn.commit()
        
        api_logger.info(json.dumps({
            'request_id': request_id,
            'status': 'success',
            'file': filename,
            'chunk_count': chunk_count
        }))
        
    except Exception as e:
        api_logger.error(json.dumps({
            'request_id': request_id,
            'status': 'error',
            'error': str(e)
        }))
        raise

def force_reprocess_files(conn: sqlite3.Connection, filenames: List[str]) -> None:
    """
    Force reprocessing of specific files by removing their records

    Args:
        conn: SQLite database connection
        filenames: List of file paths to reprocess
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-force-reprocess".encode()).hexdigest()[:8]
    
    try:
        c = conn.cursor()
        for filename in filenames:
            basename = os.path.basename(filename)
            # Delete from processed_files
            c.execute("DELETE FROM processed_files WHERE filename = ?", (basename,))
            
            # Delete from chunks
            c.execute("DELETE FROM chunks WHERE filename = ?", (basename,))
            
            # Delete from documents
            c.execute("DELETE FROM documents WHERE filename = ?", (basename,))
        
        conn.commit()
        api_logger.info(json.dumps({
            'request_id': request_id,
            'status': 'success',
            'files_cleared': len(filenames)
        }))
        
    except Exception as e:
        conn.rollback()
        api_logger.error(json.dumps({
            'request_id': request_id,
            'status': 'error',
            'error': str(e)
        }))
        raise

def get_database_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Get statistics about the database contents
    
    Args:
        conn: SQLite database connection
    
    Returns:
        Dict containing database statistics:
        - total_files: Number of files
        - total_chunks: Number of chunks
        - recent_files: List of recently processed files
        - database_size: Size of the database in bytes
    
    Example:
        stats = get_database_stats(conn)
        print(f"Database size: {stats['database_size']} bytes")
    """
    c = conn.cursor()
    stats = {}
    
    try:
        c.execute('SELECT COUNT(DISTINCT filename) FROM documents')
        stats['total_files'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM documents')
        stats['total_chunks'] = c.fetchone()[0]
        
        c.execute('''SELECT filename, processed_at 
                    FROM processed_files 
                    ORDER BY processed_at DESC 
                    LIMIT 5''')
        stats['recent_files'] = c.fetchall()
        
        c.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['database_size'] = c.fetchone()[0]
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting database stats: {str(e)}")
        raise