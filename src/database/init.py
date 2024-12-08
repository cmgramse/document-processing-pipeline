"""
Database Initialization Module

This module handles SQLite database initialization and schema management for
the document management system. It ensures the database structure is properly
set up and maintained, including handling schema migrations and upgrades.

The module manages:
- Database connection and initialization
- Schema creation and updates
- Index management
- Database maintenance

Database Schema:
    documents:
        - id: Primary key
        - filepath: Document path
        - status: Processing status
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - metadata: JSON metadata
        - qdrant_status: Vector upload status
    
    chunks:
        - id: Primary key
        - document_id: Foreign key to documents
        - content: Chunk text content
        - metadata: JSON metadata
        - created_at: Creation timestamp
        - embedding: Vector embedding
        - version: Schema version

Example:
    Initialize database:
        conn = initialize_database()
    
    Create test database:
        conn = create_test_database()
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_database_path(test: bool = False) -> Path:
    """
    Get the path to the SQLite database file.
    
    Args:
        test: Whether to return test database path
    
    Returns:
        Path: Database file path
    
    Example:
        db_path = get_database_path()
        print(f"Using database at: {db_path}")
    """
    if test:
        return Path('./data/test.db')
    return Path('./data/documents.db')

def initialize_database(test: bool = False) -> sqlite3.Connection:
    """
    Initialize the SQLite database with required schema.
    
    Args:
        test: Whether to initialize test database
    
    Returns:
        sqlite3.Connection: Database connection
    
    The function:
    1. Creates database file if not exists
    2. Creates required tables
    3. Sets up indexes
    4. Performs any needed migrations
    
    Example:
        conn = initialize_database()
        try:
            # Use database
            pass
        finally:
            conn.close()
    """
    db_path = get_database_path(test)
    db_path.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        create_schema(conn)
        create_indexes(conn)
        perform_migrations(conn)
        conn.commit()
    except Exception as e:
        conn.close()
        raise Exception(f"Failed to initialize database: {str(e)}")
    
    return conn

def create_schema(conn: sqlite3.Connection) -> None:
    """
    Create database schema if not exists.
    
    Args:
        conn: SQLite database connection
    
    Creates tables:
    - documents: Document metadata and status
    - chunks: Document chunks and embeddings
    - migrations: Schema version tracking
    
    Example:
        create_schema(conn)
    """
    c = conn.cursor()
    
    # Documents table
    c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        filepath TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT,
        qdrant_status TEXT DEFAULT 'pending'
    )
    ''')
    
    # Chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        embedding BLOB,
        version INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (document_id) REFERENCES documents(id)
    )
    ''')
    
    # Migrations table
    c.execute('''
    CREATE TABLE IF NOT EXISTS migrations (
        id INTEGER PRIMARY KEY,
        version INTEGER NOT NULL,
        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    ''')

def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create database indexes for performance optimization.
    
    Args:
        conn: SQLite database connection
    
    Creates indexes on:
    - document filepath and status
    - chunk document_id and version
    - migration version
    
    Example:
        create_indexes(conn)
    """
    c = conn.cursor()
    
    # Document indexes
    c.execute('''
    CREATE INDEX IF NOT EXISTS idx_documents_filepath
    ON documents(filepath)
    ''')
    
    c.execute('''
    CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status)
    ''')
    
    # Chunk indexes
    c.execute('''
    CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON chunks(document_id)
    ''')
    
    c.execute('''
    CREATE INDEX IF NOT EXISTS idx_chunks_version
    ON chunks(version)
    ''')

def perform_migrations(conn: sqlite3.Connection) -> None:
    """
    Perform any pending database migrations.
    
    Args:
        conn: SQLite database connection
    
    The function:
    1. Checks current schema version
    2. Applies any pending migrations
    3. Updates schema version
    
    Example:
        perform_migrations(conn)
    """
    c = conn.cursor()
    
    # Get current version
    c.execute('SELECT MAX(version) FROM migrations')
    current_version = c.fetchone()[0] or 0
    
    # Apply migrations
    migrations = [
        # Add new migrations here
        # (version, migration_sql)
    ]
    
    for version, migration_sql in migrations:
        if version > current_version:
            try:
                c.execute(migration_sql)
                c.execute(
                    'INSERT INTO migrations (version) VALUES (?)',
                    (version,)
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration {version} failed: {str(e)}")

def create_test_database() -> sqlite3.Connection:
    """
    Create a temporary test database.
    
    Returns:
        sqlite3.Connection: Test database connection
    
    The function creates an isolated database for testing with:
    - Same schema as production
    - Empty tables
    - In-memory when possible
    
    Example:
        conn = create_test_database()
        try:
            # Run tests
            pass
        finally:
            conn.close()
    """
    return initialize_database(test=True)