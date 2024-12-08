"""
Configuration Module

This module handles application configuration, logging setup, and environment
validation for the document management system. It provides centralized
configuration management and ensures all required settings are available
before the application starts.

The module manages:
- Environment variable validation
- Logging configuration
- Application settings
- API credentials

Required Environment Variables:
    JINA_API_KEY: API key for Jina AI
    QDRANT_API_KEY: API key for Qdrant
    QDRANT_URL: URL for Qdrant service
    QDRANT_COLLECTION_NAME: Name of Qdrant collection

Example:
    Set up logging:
        logger = setup_logging()
    
    Validate environment:
        check_environment()
    
    Get configuration:
        config = get_config()
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

def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up application logging with file and console handlers.
    
    Args:
        log_level: Optional logging level (defaults to config value)
    
    Returns:
        Logger: Configured logger instance
    
    The function sets up:
    - Console handler with INFO level
    - File handler with DEBUG level
    - API calls log for tracking API operations
    
    Example:
        logger = setup_logging('DEBUG')
        logger.info("Application started")
    """
    # Ensure logs directory exists
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    level = log_level or DEFAULT_CONFIG['log_level']
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'app.log', mode='w')
        ]
    )
    
    # Configure API calls logger
    api_logger = logging.getLogger('api_calls')
    api_logger.setLevel(logging.DEBUG)
    
    api_handler = logging.FileHandler(log_dir / 'api_calls.log', mode='w')
    api_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    )
    api_logger.addHandler(api_handler)
    
    return api_logger

def check_environment() -> None:
    """
    Validate required environment variables and application setup.
    
    Raises:
        EnvironmentError: If required variables are missing
    
    The function checks:
    - Required API keys
    - Service endpoints
    - Directory structure
    - Database access
    
    Example:
        try:
            check_environment()
        except EnvironmentError as e:
            print(f"Environment not properly configured: {e}")
    """
    required_vars = [
        'JINA_API_KEY',
        'QDRANT_API_KEY',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    # Ensure required directories exist
    for dir_path in ['./docs', './data', './logs']:
        Path(dir_path).mkdir(exist_ok=True)

def get_config() -> Dict[str, Any]:
    """
    Get application configuration with environment overrides.
    
    Returns:
        Dict containing configuration values
    
    The function:
    1. Starts with default configuration
    2. Applies environment variable overrides
    3. Validates configuration values
    
    Example:
        config = get_config()
        chunk_size = config['chunk_size']
    """
    config = DEFAULT_CONFIG.copy()
    
    # Override from environment variables
    env_overrides = {
        'CHUNK_SIZE': ('chunk_size', int),
        'BATCH_SIZE': ('batch_size', int),
        'RETENTION_DAYS': ('retention_days', int),
        'LOG_LEVEL': ('log_level', str),
        'DOCS_DIR': ('docs_dir', str),
        'DATABASE_PATH': ('database_path', str)
    }
    
    for env_var, (config_key, type_func) in env_overrides.items():
        if value := os.getenv(env_var):
            try:
                config[config_key] = type_func(value)
            except ValueError as e:
                logging.warning(
                    f"Invalid value for {env_var}, using default: {str(e)}"
                )
    
    return config

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

def setup_logging_original():
    """Configure detailed logging for the application"""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add specific loggers for APIs
    api_logger = logging.getLogger('api_calls')
    api_logger.setLevel(logging.INFO)
    api_handler = logging.FileHandler('logs/api_calls.log')
    api_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    api_logger.addHandler(api_handler)
    
    # Ensure API logger doesn't propagate to root logger
    api_logger.propagate = False
    
    return api_logger

def check_environment_original():
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