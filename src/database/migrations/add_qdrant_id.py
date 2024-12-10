"""
Migration script to add qdrant_id column to chunks table.
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)

def migrate(conn: sqlite3.Connection) -> None:
    """Add qdrant_id column to chunks table if it doesn't exist."""
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("SELECT qdrant_id FROM chunks LIMIT 1")
        logger.info("qdrant_id column already exists")
        return
    except sqlite3.OperationalError:
        logger.info("Adding qdrant_id column to chunks table...")
        
        # Add the column
        cursor.execute("""
            ALTER TABLE chunks 
            ADD COLUMN qdrant_id TEXT
        """)
        
        # Create index for the new column
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_id 
            ON chunks(qdrant_id)
        """)
        
        conn.commit()
        logger.info("Successfully added qdrant_id column")

if __name__ == '__main__':
    # This allows running the migration directly
    import os
    db_path = os.path.join('data', 'documents.db')
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        exit(1)
        
    conn = sqlite3.connect(db_path)
    try:
        migrate(conn)
    finally:
        conn.close() 