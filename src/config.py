import os
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

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

def setup_logging():
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