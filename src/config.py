"""
Configuration Module

This module handles application configuration, logging setup, and environment
validation for the document management system. It provides centralized
configuration management and ensures all required settings are available
before the application starts.

The module manages:
- Environment variable validation
- Application settings
- API credentials

Required Environment Variables:
    JINA_API_KEY: API key for Jina AI
    QDRANT_API_KEY: API key for Qdrant
    QDRANT_URL: URL for Qdrant service
    QDRANT_COLLECTION_NAME: Name of Qdrant collection
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

def load_environment():
    """Load environment variables with detailed validation"""
    logger = logging.getLogger('api_calls')
    
    # Try to find .env file
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise EnvironmentError("No .env file found. Please create one with required variables.")
    
    # Load .env file
    loaded = load_dotenv(dotenv_path)
    if not loaded:
        raise EnvironmentError("Failed to load .env file")
    
    logger.info(f"Successfully loaded environment from: {dotenv_path}")
    return True

def load_dotenv(dotenv_path: str) -> bool:
    """Load environment variables from .env file"""
    with open(dotenv_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            os.environ[key] = value
    return True

def find_dotenv() -> Optional[str]:
    """Find .env file in the current directory"""
    dotenv_path = './.env'
    if Path(dotenv_path).exists():
        return dotenv_path
    return None

def check_environment():
    """Validate required environment variables and directories"""
    logger = logging.getLogger('api_calls')
    
    # First, ensure environment is loaded
    load_environment()
    
    # Check required variables
    required_vars = {
        "JINA_API_KEY": "Required for document segmentation and embeddings",
        "QDRANT_API_KEY": "Required for vector database operations",
        "QDRANT_URL": "Required for connecting to Qdrant instance",
        "QDRANT_COLLECTION_NAME": "Required for storing vectors"
    }
    
    missing_vars = []
    empty_vars = []
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value is None:
            missing_vars.append(f"{var} ({description})")
        elif not value.strip():
            empty_vars.append(f"{var} ({description})")
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables:\n" + "\n".join(missing_vars))
    
    if empty_vars:
        raise EnvironmentError(f"Following environment variables are empty:\n" + "\n".join(empty_vars))
    
    # Log successful validation of each variable
    for var in required_vars:
        logger.info(f" Validated {var}")
    
    # Check docs directory
    docs_path = Path('./docs')
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logger.info("Created docs directory")
    else:
        logger.info(" Docs directory exists")
    
    logger.info("Environment validation complete - all checks passed")
    return True