"""
SQLAlchemy session management.
"""

import os
import logging
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from .models import Base, Chunk, Document

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DB_PATH = os.getenv('DB_PATH', 'data/documents.db')
db_dir = Path(DB_PATH).parent
db_dir.mkdir(exist_ok=True)

# Get pool configuration from environment
POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))
POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', '1800'))

# Create engine with proper configuration
engine = create_engine(
    f'sqlite:///{DB_PATH}',
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    connect_args={
        'timeout': POOL_TIMEOUT,
        'check_same_thread': False  # Required for SQLite
    }
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

def check_database_state() -> dict:
    """
    Check the current state of the database.
    
    Returns:
        Dict containing database state information
    """
    with get_db() as session:
        # Check if tables exist
        inspector = inspect(engine)
        tables_exist = all(table in inspector.get_table_names() 
                         for table in ['documents', 'chunks', 'processed_files'])
        
        if not tables_exist:
            return {'needs_init': True, 'reason': 'Tables missing'}
            
        # Check for pending tasks
        pending_embeddings = session.query(Chunk).filter_by(embedding_status='pending').count()
        pending_qdrant = session.query(Chunk).filter_by(qdrant_status='pending').count()
        
        return {
            'needs_init': False,
            'pending_embeddings': pending_embeddings,
            'pending_qdrant': pending_qdrant,
            'total_documents': session.query(Document).count(),
            'total_chunks': session.query(Chunk).count()
        }

def init_db(drop_all: bool = False) -> None:
    """Initialize the database, optionally dropping all tables first."""
    db_state = check_database_state()
    
    if drop_all or db_state.get('needs_init', False):
        logger.warning("Database schema incomplete, reinitializing...")
        if drop_all:
            logger.info("Dropping all tables...")
            Base.metadata.drop_all(bind=engine)
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema restored")
    else:
        pending = db_state.get('pending_embeddings', 0) + db_state.get('pending_qdrant', 0)
        if pending > 0:
            logger.info(f"Found {pending} pending tasks in database")
            logger.info(f"- Pending embeddings: {db_state['pending_embeddings']}")
            logger.info(f"- Pending Qdrant uploads: {db_state['pending_qdrant']}")

@contextmanager
def get_db():
    """Get a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal() 

def ensure_database() -> bool:
    """Ensure database exists and is properly initialized."""
    # Get database path from environment
    db_path = Path(os.getenv('DB_PATH', 'data/documents.db'))
    db_dir = db_path.parent
    db_dir.mkdir(exist_ok=True)
    
    needs_init = not db_path.exists()
    
    if needs_init:
        logger.info("Initializing new database...")
        init_db(drop_all=False)
        logger.info(f"Database initialized at {db_path}")
    
    # Verify database
    with get_db() as session:
        try:
            # Test database connection using SQLAlchemy text()
            from sqlalchemy import text
            session.execute(text("SELECT 1"))
            if needs_init:
                logger.info("Database connection verified")
            
            # Verify schema by checking for required tables
            tables = session.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND 
                name IN ('document', 'chunk')
            """)).fetchall()
            
            if len(tables) < 2:
                logger.warning("Database schema incomplete, reinitializing...")
                init_db(drop_all=True)
                logger.info("Database schema restored")
            
            return True
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            try:
                logger.info("Attempting database recovery...")
                init_db(drop_all=True)
                logger.info("Database recovered successfully")
                return True
            except Exception as e:
                logger.error(f"Database recovery failed: {e}")
                return False 