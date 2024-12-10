"""
Main application entry point.
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .database.connection import init_connection_pool, close_connection_pool
from .database.document_manager import DocumentManager
from .api.routes import router

logger = logging.getLogger(__name__)

# Configure database path
DB_PATH = os.getenv('DB_PATH', 'data/documents.db')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    Initialize database connection pool and clean up resources.
    """
    # Ensure data directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize connection pool
    init_connection_pool(DB_PATH, max_connections=5)
    
    # Initialize document manager (creates tables)
    DocumentManager()
    
    logger.info("Application startup complete")
    yield
    
    # Cleanup on shutdown
    close_connection_pool()
    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Document Processing API",
    description="API for document processing and vector search",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(router) 