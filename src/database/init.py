"""
Database initialization and setup module.
"""

import sqlite3
import logging
from pathlib import Path
import os

def init_database(db_path: str) -> None:
    """
    Initialize the SQLite database with proper schema.
    
    Args:
        db_path: Path to the database file
    """
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set journal mode to WAL for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Set synchronous mode to NORMAL for better performance while maintaining safety
        conn.execute("PRAGMA synchronous = NORMAL")
        
        # Read and execute schema
        schema_path = Path(__file__).parent / 'schema.sql'
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        # Split schema into individual statements and execute each one
        statements = schema.split(';')
        for statement in statements:
            if statement.strip():
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        raise
                        
        conn.commit()
        logging.info(f"Database initialized at {db_path}")
        
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a database connection with proper settings.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        SQLite connection object
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Set row factory for better column access
        conn.row_factory = sqlite3.Row
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set journal mode to WAL for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Set synchronous mode to NORMAL for better performance while maintaining safety
        conn.execute("PRAGMA synchronous = NORMAL")
        
        return conn
        
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        raise

def check_database(db_path: str) -> bool:
    """
    Check if database exists and has proper schema.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if database is valid
    """
    if not os.path.exists(db_path):
        return False
        
    try:
        conn = get_connection(db_path)
        c = conn.cursor()
        
        # Check for required tables
        required_tables = {
            'documents',
            'chunks',
            'processed_files',
            'document_references',
            'processing_queue',
            'processing_history'
        }
        
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in c.fetchall()}
        
        # Check if all required tables exist
        missing_tables = required_tables - existing_tables
        if missing_tables:
            logging.error(f"Missing tables: {missing_tables}")
            return False
            
        # Check for required columns in chunks table
        c.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in c.fetchall()}
        
        required_columns = {
            'id',
            'filename',
            'content',
            'token_count',
            'chunk_number',
            'content_hash',
            'chunking_status',
            'embedding_status',
            'qdrant_status',
            'embedding',
            'qdrant_id',
            'processed_at',
            'created_at',
            'last_verified_at',
            'error_message',
            'version'
        }
        
        missing_columns = required_columns - columns
        if missing_columns:
            logging.error(f"Missing columns in chunks table: {missing_columns}")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error checking database: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def initialize_if_needed(db_path: str) -> None:
    """
    Initialize database if it doesn't exist or is invalid.
    
    Args:
        db_path: Path to the database file
    """
    try:
        if not check_database(db_path):
            logging.info(f"Initializing database at {db_path}")
            init_database(db_path)
        else:
            logging.info(f"Database at {db_path} is valid")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise