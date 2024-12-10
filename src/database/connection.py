"""
Database connection management module.

Handles database connections, retries, and maintenance.
"""

import logging
import sqlite3
import time
from typing import Optional, Generator, Any
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
from pathlib import Path

from ..config.settings import settings
from ..processing.error_handler import DatabaseError, handle_error

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and maintenance."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize database manager."""
        if not self._initialized:
            self.db_path = Path(settings.db.path)
            self._connections = {}
            self._last_vacuum = None
            self._vacuum_interval = timedelta(days=7)
            self._initialized = True
    
    @contextmanager
    def get_connection(self, max_retries: int = 3, retry_delay: int = 1) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
            
        Yields:
            Database connection
        """
        conn = None
        try:
            for attempt in range(max_retries):
                try:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=settings.db.pool_timeout
                    )
                    
                    # Enable foreign keys
                    conn.execute("PRAGMA foreign_keys = ON")
                    
                    # Set journal mode to WAL for better concurrency
                    conn.execute("PRAGMA journal_mode = WAL")
                    
                    # Set busy timeout
                    conn.execute(f"PRAGMA busy_timeout = {settings.db.pool_timeout * 1000}")
                    
                    yield conn
                    conn.commit()  # Commit any pending changes
                    return
                    
                except sqlite3.Error as e:
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    
                    if attempt == max_retries - 1:
                        raise DatabaseError(f"Failed to connect to database: {e}")
                    
                    logger.warning(
                        f"Database connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            try:
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error closing database connection: {e}")
        self._connections.clear()
    
    def vacuum_if_needed(self) -> None:
        """Run VACUUM if needed."""
        now = datetime.now()
        
        # Check if vacuum is needed
        if (self._last_vacuum is None or 
            now - self._last_vacuum > self._vacuum_interval):
            
            try:
                # Create temporary connection for vacuum
                conn = sqlite3.connect(str(self.db_path))
                try:
                    # Get database size before vacuum
                    size_before = self.db_path.stat().st_size / (1024 * 1024)  # MB
                    
                    # Run vacuum
                    logger.info("Running database VACUUM...")
                    conn.execute("VACUUM")
                    conn.commit()
                    
                    # Get database size after vacuum
                    size_after = self.db_path.stat().st_size / (1024 * 1024)  # MB
                    
                    # Log results
                    space_saved = size_before - size_after
                    logger.info(
                        f"Database VACUUM complete. "
                        f"Size before: {size_before:.1f}MB, "
                        f"Size after: {size_after:.1f}MB, "
                        f"Space saved: {space_saved:.1f}MB"
                    )
                    
                    self._last_vacuum = now
                    
                finally:
                    conn.close()
                    
            except sqlite3.Error as e:
                logger.error(f"Error running database VACUUM: {e}")
    
    def optimize(self) -> None:
        """Run database optimization."""
        try:
            with self.get_connection() as conn:
                # Run ANALYZE to update statistics
                conn.execute("ANALYZE")
                
                # Optimize indexes
                conn.execute("PRAGMA optimize")
                
                # Clean up WAL file if it's too large
                wal_path = Path(str(self.db_path) + "-wal")
                if wal_path.exists():
                    wal_size = wal_path.stat().st_size / (1024 * 1024)
                    if wal_size > 100:  # If WAL is larger than 100MB
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                
                conn.commit()
                
                logger.info("Database optimization complete")
                
        except sqlite3.Error as e:
            logger.error(f"Error optimizing database: {e}")
    
    @handle_error('database')
    def check_integrity(self) -> bool:
        """
        Check database integrity.
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Check integrity
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                
                # Check foreign keys
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                
                if result != "ok" or fk_violations:
                    logger.error(
                        f"Database integrity check failed: {result}, "
                        f"Foreign key violations: {fk_violations}"
                    )
                    return False
                
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error checking database integrity: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 