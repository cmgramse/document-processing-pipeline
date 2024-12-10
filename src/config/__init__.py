"""
Configuration package initialization.

This module exposes the main configuration functionality.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    'chunk_size': 1000,
    'batch_size': 50,
    'retention_days': 30,
    'log_level': 'INFO',
    'docs_dir': './docs',
    'database_path': './data/documents.db',
}

def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Set up application logging with file and console handlers.
    
    Args:
        log_level: Optional logging level (defaults to config value)
    
    The function sets up:
    - Console handler with INFO level
    - File handler with DEBUG level in /logs directory
    - API calls log for tracking API operations
    - Error log for tracking errors
    - Pipeline log for document processing
    - Metrics log for performance tracking
    """
    # Ensure logs directory exists with proper permissions
    log_dir = Path('./logs')
    log_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    
    # Configure root logger
    level = getattr(logging, log_level or DEFAULT_CONFIG['log_level'])
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'app.log', mode='a')
        ]
    )
    
    # Configure API calls logger
    api_logger = logging.getLogger('api_calls')
    api_logger.setLevel(logging.DEBUG)
    api_handler = logging.FileHandler(log_dir / 'api_calls.log', mode='a')
    api_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    api_logger.addHandler(api_handler)
    api_logger.propagate = False
    
    # Configure error logger
    error_logger = logging.getLogger('errors')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(log_dir / 'errors.log', mode='a')
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s\nTraceback:\n%(exc_info)s')
    )
    error_logger.addHandler(error_handler)
    error_logger.propagate = False
    
    # Configure pipeline logger
    pipeline_logger = logging.getLogger('pipeline')
    pipeline_logger.setLevel(logging.INFO)
    pipeline_handler = logging.FileHandler(log_dir / 'pipeline.log', mode='a')
    pipeline_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    pipeline_logger.addHandler(pipeline_handler)
    pipeline_logger.propagate = False
    
    # Configure metrics logger
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.setLevel(logging.INFO)
    metrics_handler = logging.FileHandler(log_dir / 'metrics.log', mode='a')
    metrics_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    metrics_logger.addHandler(metrics_handler)
    metrics_logger.propagate = False
