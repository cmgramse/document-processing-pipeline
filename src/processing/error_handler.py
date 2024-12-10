"""
Error handling module.

Provides consistent error handling and reporting across the application.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Type, Callable, List
from functools import wraps
from datetime import datetime
import sqlite3
from pathlib import Path
import json
import asyncio

from ..config.settings import settings

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Base exception for processing errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()

class ValidationError(ProcessingError):
    """Raised when document validation fails."""
    pass

class ResourceError(ProcessingError):
    """Raised when resource constraints are violated."""
    pass

class APIError(ProcessingError):
    """Raised when external API calls fail."""
    pass

class DatabaseError(ProcessingError):
    """Raised when database operations fail."""
    pass

class QueueError(ProcessingError):
    """Raised when queue operations fail."""
    pass

class ErrorStore:
    """Stores and manages error records."""
    
    def __init__(self):
        data_dir = Path('./data')
        data_dir.mkdir(exist_ok=True)
        self.db_path = data_dir / "errors.db"
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize error database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                traceback TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_note TEXT,
                resolution_timestamp TIMESTAMP
            )
            """)
            
            # Create indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_errors_resolved ON errors(resolved)")
            
            conn.commit()
        finally:
            conn.close()
    
    def record_error(
        self,
        error: Exception,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record error in database.
        
        Args:
            error: The exception that occurred
            error_type: Type of error (e.g., 'validation', 'api', 'database')
            details: Additional error details
            
        Returns:
            Error ID
            
        Raises:
            RuntimeError: If error could not be recorded
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO errors (
                    timestamp, error_type, message, details, traceback
                ) VALUES (datetime('now'), ?, ?, ?, ?)
                """,
                (
                    error_type,
                    str(error),
                    json.dumps(details) if details else None,
                    ''.join(traceback.format_tb(error.__traceback__))
                )
            )
            conn.commit()
            if c.lastrowid is None:
                raise RuntimeError("Failed to get last inserted row ID")
            return c.lastrowid
        finally:
            conn.close()
    
    def resolve_error(self, error_id: int, resolution_note: str) -> None:
        """Mark error as resolved with note."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute(
                """
                UPDATE errors
                SET resolved = TRUE,
                    resolution_note = ?,
                    resolution_timestamp = datetime('now')
                WHERE id = ?
                """,
                (resolution_note, error_id)
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_error(self, error_id: int) -> Optional[Dict[str, Any]]:
        """Get error record by ID."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute(
                """
                SELECT id, timestamp, error_type, message, details,
                       traceback, resolved, resolution_note,
                       resolution_timestamp
                FROM errors WHERE id = ?
                """,
                (error_id,)
            )
            row = c.fetchone()
            if row:
                return {
                    'id': row[0],
                    'timestamp': row[1],
                    'error_type': row[2],
                    'message': row[3],
                    'details': json.loads(row[4]) if row[4] else None,
                    'traceback': row[5],
                    'resolved': bool(row[6]),
                    'resolution_note': row[7],
                    'resolution_timestamp': row[8]
                }
            return None
        finally:
            conn.close()
    
    def get_unresolved_errors(
        self,
        error_type: Optional[str] = None,
        limit: int = 100
    ) -> 'List[Dict[str, Any]]':
        """Get unresolved errors."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            query = """
                SELECT id, timestamp, error_type, message, details
                FROM errors
                WHERE resolved = FALSE
            """
            params = []
            
            if error_type:
                query += " AND error_type = ?"
                params.append(error_type)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            c.execute(query, params)
            return [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'error_type': row[2],
                    'message': row[3],
                    'details': json.loads(row[4]) if row[4] else None
                }
                for row in c.fetchall()
            ]
        finally:
            conn.close()

# Global error store instance
error_store = ErrorStore()

def handle_error(
    error_type: str,
    reraise: bool = True,
    log_level: int = logging.ERROR
) -> Callable:
    """
    Decorator for consistent error handling.
    
    Args:
        error_type: Type of error for categorization
        reraise: Whether to reraise the error after handling
        log_level: Logging level for error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error details
                details = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                if isinstance(e, ProcessingError):
                    details.update(e.details)
                
                # Record error
                error_id = error_store.record_error(e, error_type, details)
                
                # Log error
                logger.log(
                    log_level,
                    f"Error in {func.__name__} (ID: {error_id}): {str(e)}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                
                return None
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error details
                details = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                if isinstance(e, ProcessingError):
                    details.update(e.details)
                
                # Record error
                error_id = error_store.record_error(e, error_type, details)
                
                # Log error
                logger.log(
                    log_level,
                    f"Error in {func.__name__} (ID: {error_id}): {str(e)}",
                    exc_info=True
                )
                
                if reraise:
                    raise
                
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def with_error_handling(
    error_mapping: Dict[Type[Exception], Type[ProcessingError]],
    error_type: str,
    reraise: bool = True
) -> Callable:
    """
    Decorator for mapping exceptions to custom error types.
    
    Args:
        error_mapping: Mapping of exception types to custom error types
        error_type: Type of error for categorization
        reraise: Whether to reraise the error after handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                for exc_type, error_class in error_mapping.items():
                    if isinstance(e, exc_type):
                        raise error_class(str(e)) from e
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                for exc_type, error_class in error_mapping.items():
                    if isinstance(e, exc_type):
                        raise error_class(str(e)) from e
                raise
        
        wrapped = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return handle_error(error_type, reraise)(wrapped)
    
    return decorator

# Example usage:
"""
@handle_error('validation')
def validate_document(document: Document) -> bool:
    # Validation logic here
    pass

@with_error_handling(
    {
        requests.RequestException: APIError,
        ValueError: ValidationError
    },
    'api'
)
async def call_external_api(data: Dict[str, Any]) -> Dict[str, Any]:
    # API call logic here
    pass
""" 