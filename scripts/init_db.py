"""
Database Initialization Script

This script initializes the SQLite database with the schema defined in schema.sql.
It also ensures all required directories exist.

Usage:
    python -m scripts.init_db
"""

import os
import logging
from pathlib import Path
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories() -> None:
    """Ensure all required directories exist."""
    dirs = [
        './data',    # For SQLite database
        './logs',    # For log files
        './docs'     # For document storage
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Created directory: {dir_path}")

def init_database() -> None:
    """Initialize the SQLite database with schema."""
    try:
        # Ensure data directory exists
        ensure_directories()
        
        # Connect to database
        db_path = Path('./data/documents.db')
        conn = sqlite3.connect(db_path)
        
        # Read schema
        schema_path = Path('./src/database/schema.sql')
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        # Execute schema
        conn.executescript(schema)
        conn.commit()
        
        logger.info(f"Initialized database at: {db_path}")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def main() -> None:
    """Main initialization function."""
    try:
        # Initialize database
        init_database()
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 