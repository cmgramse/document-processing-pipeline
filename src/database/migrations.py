"""
Database migrations module.

Handles database schema migrations in a versioned way.
"""

import logging
import sqlite3
import json
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import hashlib
import fcntl
import time

from ..models.document import ProcessingStatus

logger = logging.getLogger(__name__)

class MigrationError(Exception):
    """Base exception for migration errors."""
    pass

class MigrationLockError(MigrationError):
    """Raised when migration lock cannot be acquired."""
    pass

class MigrationStateError(MigrationError):
    """Raised when migration state is invalid."""
    pass

class Migration:
    """Base class for database migrations."""
    
    def __init__(self, version: int, description: str):
        self.version = version
        self.description = description
        self._validate()
    
    def _validate(self) -> None:
        """Validate migration attributes."""
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("Version must be a positive integer")
        if not self.description:
            raise ValueError("Description is required")
    
    def up(self, conn: sqlite3.Connection) -> None:
        """Apply migration."""
        raise NotImplementedError
    
    def down(self, conn: sqlite3.Connection) -> None:
        """Revert migration."""
        raise NotImplementedError
    
    def get_checksum(self) -> str:
        """Get migration checksum for validation."""
        content = f"{self.version}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()

class MigrationManager:
    """Manages database migrations with locking and validation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock_path = Path(db_path).parent / ".migration.lock"
        self.lock_file = None
    
    def __enter__(self):
        """Acquire migration lock."""
        try:
            self.lock_file = open(self.lock_path, 'w')
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self
        except (IOError, OSError) as e:
            if self.lock_file:
                self.lock_file.close()
            raise MigrationLockError("Migration already in progress") from e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release migration lock."""
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_path.unlink(missing_ok=True)
    
    def init_migration_tables(self, conn: sqlite3.Connection) -> None:
        """Initialize migration tracking tables."""
        c = conn.cursor()
        
        # Migrations table - tracks applied migrations
        c.execute("""
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            description TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time REAL NOT NULL,
            batch INTEGER NOT NULL
        )
        """)
        
        # Migration history - tracks all migration attempts
        c.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            execution_time REAL
        )
        """)
        
        # Create indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_migrations_version ON migrations(version)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_migration_history_version ON migration_history(version)")
    
    def get_applied_migrations(self, conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
        """Get all applied migrations."""
        c = conn.cursor()
        c.execute("SELECT version, description, checksum, applied_at FROM migrations ORDER BY version")
        return {
            row[0]: {
                'description': row[1],
                'checksum': row[2],
                'applied_at': row[3]
            }
            for row in c.fetchall()
        }
    
    def validate_migrations(self, conn: sqlite3.Connection, migrations: List[Migration]) -> None:
        """Validate migration state and checksums."""
        applied = self.get_applied_migrations(conn)
        
        # Check for missing migrations
        available = {m.version: m for m in migrations}
        for version in applied:
            if version not in available:
                raise MigrationStateError(f"Migration version {version} is applied but not available")
        
        # Validate checksums
        for version, migration in available.items():
            if version in applied:
                applied_checksum = applied[version]['checksum']
                current_checksum = migration.get_checksum()
                if applied_checksum != current_checksum:
                    raise MigrationStateError(
                        f"Checksum mismatch for version {version}. "
                        f"Applied: {applied_checksum}, Current: {current_checksum}"
                    )
    
    def record_migration_attempt(
        self,
        conn: sqlite3.Connection,
        version: int,
        operation: str,
        status: str,
        error: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """Record migration attempt in history."""
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO migration_history (
                version, operation, status, error_message,
                started_at, completed_at, execution_time
            ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), ?)
            """,
            (version, operation, status, error, execution_time)
        )
    
    def migrate(
        self,
        conn: sqlite3.Connection,
        migrations: List[Migration],
        target_version: Optional[int] = None
    ) -> None:
        """
        Run database migrations with validation and history tracking.
        
        Args:
            conn: Database connection
            migrations: List of available migrations
            target_version: Optional target version, defaults to latest
        """
        # Initialize tables
        self.init_migration_tables(conn)
        
        # Validate migrations
        self.validate_migrations(conn, migrations)
        
        # Get current state
        current_version = get_current_version(conn)
        migrations_dict = {m.version: m for m in migrations}
        
        if target_version is None:
            target_version = max(m.version for m in migrations)
        
        try:
            if target_version > current_version:
                # Get next batch number
                c = conn.cursor()
                c.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM migrations")
                batch = c.fetchone()[0]
                
                # Migrate up
                for version in range(current_version + 1, target_version + 1):
                    if version not in migrations_dict:
                        raise MigrationStateError(f"Missing migration version {version}")
                    
                    migration = migrations_dict[version]
                    start_time = time.time()
                    
                    try:
                        # Record attempt
                        self.record_migration_attempt(
                            conn, version, "up", "in_progress"
                        )
                        
                        # Apply migration
                        migration.up(conn)
                        
                        # Calculate execution time
                        execution_time = time.time() - start_time
                        
                        # Record successful migration
                        c.execute(
                            """
                            INSERT INTO migrations (
                                version, description, checksum,
                                applied_at, execution_time, batch
                            ) VALUES (?, ?, ?, datetime('now'), ?, ?)
                            """,
                            (
                                version,
                                migration.description,
                                migration.get_checksum(),
                                execution_time,
                                batch
                            )
                        )
                        
                        # Update history
                        self.record_migration_attempt(
                            conn, version, "up", "completed",
                            execution_time=execution_time
                        )
                        
                        conn.commit()
                        logger.info(
                            f"Applied migration {version} ({execution_time:.2f}s): "
                            f"{migration.description}"
                        )
                        
                    except Exception as e:
                        conn.rollback()
                        self.record_migration_attempt(
                            conn, version, "up", "failed",
                            error=str(e)
                        )
                        raise MigrationError(
                            f"Failed to apply migration {version}: {e}"
                        ) from e
                        
            elif target_version < current_version:
                # Migrate down
                for version in range(current_version, target_version, -1):
                    if version not in migrations_dict:
                        raise MigrationStateError(f"Missing migration version {version}")
                    
                    migration = migrations_dict[version]
                    start_time = time.time()
                    
                    try:
                        # Record attempt
                        self.record_migration_attempt(
                            conn, version, "down", "in_progress"
                        )
                        
                        # Revert migration
                        migration.down(conn)
                        
                        # Calculate execution time
                        execution_time = time.time() - start_time
                        
                        # Remove migration record
                        c.execute(
                            "DELETE FROM migrations WHERE version = ?",
                            (version,)
                        )
                        
                        # Update history
                        self.record_migration_attempt(
                            conn, version, "down", "completed",
                            execution_time=execution_time
                        )
                        
                        conn.commit()
                        logger.info(
                            f"Reverted migration {version} ({execution_time:.2f}s): "
                            f"{migration.description}"
                        )
                        
                    except Exception as e:
                        conn.rollback()
                        self.record_migration_attempt(
                            conn, version, "down", "failed",
                            error=str(e)
                        )
                        raise MigrationError(
                            f"Failed to revert migration {version}: {e}"
                        ) from e
                        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

class InitialSchema(Migration):
    """Initial database schema migration."""
    
    def __init__(self):
        super().__init__(1, "Initial schema creation")
    
    def up(self, conn: sqlite3.Connection) -> None:
        """Create initial schema."""
        c = conn.cursor()
        
        # Enable foreign key constraints
        c.execute("PRAGMA foreign_keys = ON")
        
        # Documents table
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            vector_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT '{ProcessingStatus.PENDING.value}'
                {ProcessingStatus.get_database_check_constraint()},
            error_message TEXT
        )
        """)
        
        # Document chunks table
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT '{ProcessingStatus.PENDING.value}'
                {ProcessingStatus.get_database_check_constraint()},
            error_message TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            UNIQUE (document_id, chunk_index)
        )
        """)
        
        # Create indexes
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_status 
        ON documents(processing_status)
        """)
        
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_vector 
        ON documents(vector_id)
        """)
        
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document 
        ON document_chunks(document_id)
        """)
        
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_status 
        ON document_chunks(processing_status)
        """)
    
    def down(self, conn: sqlite3.Connection) -> None:
        """Revert initial schema."""
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS document_chunks")
        c.execute("DROP TABLE IF EXISTS documents")

class AddProcessingStats(Migration):
    """Add processing statistics tables."""
    
    def __init__(self):
        super().__init__(2, "Add processing statistics")
    
    def up(self, conn: sqlite3.Connection) -> None:
        """Create processing stats tables."""
        c = conn.cursor()
        
        # Processing stats table
        c.execute("""
        CREATE TABLE IF NOT EXISTS processing_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            processing_time REAL NOT NULL,
            tokens_processed INTEGER NOT NULL,
            chunks_created INTEGER NOT NULL,
            total_documents INTEGER NOT NULL DEFAULT 0,
            processed_documents INTEGER NOT NULL DEFAULT 0,
            failed_documents INTEGER NOT NULL DEFAULT 0,
            skipped_documents INTEGER NOT NULL DEFAULT 0,
            total_chunks INTEGER NOT NULL DEFAULT 0,
            completed_chunks INTEGER NOT NULL DEFAULT 0,
            failed_chunks INTEGER NOT NULL DEFAULT 0,
            total_embeddings INTEGER NOT NULL DEFAULT 0,
            failed_embeddings INTEGER NOT NULL DEFAULT 0,
            system_memory_percent REAL NOT NULL DEFAULT 0,
            process_memory_mb REAL NOT NULL DEFAULT 0,
            batch_sizes TEXT,  -- JSON array of batch sizes
            errors TEXT,  -- JSON array of error messages
            retries INTEGER NOT NULL DEFAULT 0,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """)
        
        # Create indexes
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_stats_document 
        ON processing_stats(document_id)
        """)
        
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_stats_timestamp 
        ON processing_stats(timestamp)
        """)
    
    def down(self, conn: sqlite3.Connection) -> None:
        """Revert processing stats tables."""
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS processing_stats")

class AddBatchProcessing(Migration):
    """Add batch processing tables."""
    
    def __init__(self):
        super().__init__(3, "Add batch processing")
    
    def up(self, conn: sqlite3.Connection) -> None:
        """Create batch processing tables."""
        c = conn.cursor()
        
        # Processing batches table
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS processing_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_size INTEGER NOT NULL,
            start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            status TEXT DEFAULT '{ProcessingStatus.PENDING.value}'
                {ProcessingStatus.get_batch_check_constraint()},
            error_message TEXT,
            system_memory_percent REAL,
            process_memory_mb REAL
        )
        """)
        
        # Batch documents table
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS batch_documents (
            batch_id INTEGER NOT NULL,
            document_id TEXT NOT NULL,
            processing_order INTEGER NOT NULL,
            status TEXT DEFAULT '{ProcessingStatus.PENDING.value}'
                {ProcessingStatus.get_batch_check_constraint()},
            error_message TEXT,
            PRIMARY KEY (batch_id, document_id),
            FOREIGN KEY (batch_id) REFERENCES processing_batches(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """)
        
        # Create indexes
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_batches_status 
        ON processing_batches(status)
        """)
        
        c.execute("""
        CREATE INDEX IF NOT EXISTS idx_batch_docs_status 
        ON batch_documents(status)
        """)
    
    def down(self, conn: sqlite3.Connection) -> None:
        """Revert batch processing tables."""
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS batch_documents")
        c.execute("DROP TABLE IF EXISTS processing_batches")

def get_migrations() -> List[Migration]:
    """Get all available migrations in order."""
    return [
        InitialSchema(),
        AddProcessingStats(),
        AddBatchProcessing()
    ]

def get_current_version(conn: sqlite3.Connection) -> int:
    """Get current database version."""
    try:
        c = conn.cursor()
        c.execute("SELECT MAX(version) FROM migrations")
        result = c.fetchone()
        return result[0] or 0
    except sqlite3.OperationalError:
        return 0

def migrate(conn: sqlite3.Connection, target_version: Optional[int] = None) -> None:
    """
    Run database migrations.
    
    Args:
        conn: Database connection
        target_version: Optional target version, defaults to latest
    """
    with MigrationManager(conn.filename) as manager:
        manager.migrate(conn, get_migrations(), target_version)