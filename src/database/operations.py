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

def get_unprocessed_files(available_docs: List[str], conn: sqlite3.Connection) -> List[str]:
    """
    Filter out already processed files that haven't been modified since
    
    Args:
        available_docs: List of available document paths
        conn: SQLite database connection
    
    Returns:
        List of unprocessed file paths
    
    Example:
        unprocessed = get_unprocessed_files(['docs/example.md'], conn)
        print(f"Unprocessed files: {unprocessed}")
    """
    c = conn.cursor()
    unprocessed_files = []
    
    for doc in available_docs:
        file_path = Path('./docs') / doc
        logging.info(f"Checking file: {file_path}")
        
        try:
            last_modified = file_path.stat().st_mtime
            logging.info(f"Last modified time: {last_modified}")
            
            # Check if file has been processed and not modified since
            c.execute('''SELECT last_modified FROM processed_files 
                        WHERE filename = ?''', (str(doc),))
            result = c.fetchone()
            
            if not result:
                logging.info(f"File not found in processed_files: {doc}")
                unprocessed_files.append(doc)
            elif result[0] < last_modified:
                logging.info(f"File modified since last processing: {doc}")
                unprocessed_files.append(doc)
            else:
                logging.info(f"File already processed and unchanged: {doc}")
        
        except Exception as e:
            logging.error(f"Error checking file {doc}: {str(e)}")
            # If there's an error checking the file, we'll try to process it
            unprocessed_files.append(doc)
    
    return unprocessed_files

def mark_file_as_processed(filename: str, conn: sqlite3.Connection):
    """
    Mark a file as processed with its current modification time
    
    Args:
        filename: Path to the document file
        conn: SQLite database connection
    
    Example:
        mark_file_as_processed('docs/example.md', conn)
    """
    c = conn.cursor()
    file_path = Path('./docs') / filename
    last_modified = file_path.stat().st_mtime
    
    c.execute('''INSERT OR REPLACE INTO processed_files 
                (filename, last_modified, processed_at)
                VALUES (?, ?, datetime('now'))''',
             (str(filename), last_modified))
    conn.commit()

def force_reprocess_files(filenames: List[str], conn: sqlite3.Connection) -> None:
    """
    Force reprocessing of specific files by removing their records
    
    Args:
        filenames: List of file paths to reprocess
        conn: SQLite database connection
    
    Example:
        force_reprocess_files(['docs/example.md'], conn)
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-force-reprocess".encode()).hexdigest()[:8]
    
    api_logger.info(
        f"[{request_id}] Force reprocessing initiated for {len(filenames)} files"
    )
    
    c = conn.cursor()
    
    try:
        for filename in filenames:
            c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
            c.execute('DELETE FROM processed_files WHERE filename = ?', (filename,))
            api_logger.info(f"[{request_id}] Cleared records for {filename}")
        
        conn.commit()
        api_logger.info(
            f"[{request_id}] Successfully marked {len(filenames)} files for reprocessing"
        )
        
    except Exception as e:
        conn.rollback()
        error_details = {
            "error": str(e),
            "files": filenames
        }
        api_logger.error(
            f"[{request_id}] Failed to force reprocess files: {json.dumps(error_details)}"
        )
        raise Exception(f"Error marking files for reprocessing: {str(e)}") 

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