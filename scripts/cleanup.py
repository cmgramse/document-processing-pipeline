"""
Cleanup Script

This script provides utilities to clean up all data from:
- Qdrant vector database
- SQLite document database
- Log files

Usage:
    python -m scripts.cleanup [--force]
"""

import os
import sys
import json
import logging
import sqlite3
import requests
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_qdrant_collection() -> bool:
    """Delete the Qdrant collection."""
    try:
        # Get Qdrant configuration
        qdrant_url = os.getenv('QDRANT_URL')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME')
        api_key = os.getenv('QDRANT_API_KEY')
        
        if not all([qdrant_url, collection_name, api_key]):
            logger.error("Missing Qdrant configuration")
            return False
            
        # Delete collection
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.delete(
            f"{qdrant_url}/collections/{collection_name}",
            headers=headers
        )
        
        if response.status_code in [200, 404]:  # Success or already deleted
            logger.info(f"Deleted Qdrant collection: {collection_name}")
            return True
        else:
            logger.error(f"Failed to delete Qdrant collection: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting Qdrant collection: {str(e)}")
        return False

def recreate_qdrant_collection() -> bool:
    """Recreate the Qdrant collection with proper settings."""
    try:
        # Get Qdrant configuration
        qdrant_url = os.getenv('QDRANT_URL')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME')
        api_key = os.getenv('QDRANT_API_KEY')
        
        if not all([qdrant_url, collection_name, api_key]):
            logger.error("Missing Qdrant configuration")
            return False
            
        # Create collection
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Collection configuration
        config = {
            "vectors": {
                "size": 1024,  # Jina embedding dimension
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1,
            "write_consistency_factor": 1
        }
        
        response = requests.put(
            f"{qdrant_url}/collections/{collection_name}",
            headers=headers,
            json=config
        )
        
        if response.status_code == 200:
            logger.info(f"Created Qdrant collection: {collection_name}")
            return True
        else:
            logger.error(f"Failed to create Qdrant collection: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {str(e)}")
        return False

def delete_sqlite_database() -> bool:
    """Delete the SQLite database file."""
    try:
        db_path = Path('./data/documents.db')
        if db_path.exists():
            db_path.unlink()
            logger.info(f"Deleted SQLite database: {db_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting SQLite database: {str(e)}")
        return False

def clear_logs() -> bool:
    """Clear all log files."""
    try:
        log_dir = Path('./logs')
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                log_file.unlink()
            logger.info("Cleared all log files")
        return True
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        return False

def main(force: bool = False) -> None:
    """
    Main cleanup function.
    
    Args:
        force: If True, skip confirmation
    """
    if not force:
        confirm = input(
            "\nWARNING: This will delete all data from:\n"
            "- Qdrant vector database\n"
            "- SQLite document database\n"
            "- Log files\n\n"
            "Are you sure? [y/N]: "
        ).lower()
        
        if confirm not in ['y', 'yes']:
            logger.info("Cleanup cancelled")
            return
    
    # Delete Qdrant collection
    if delete_qdrant_collection():
        # Recreate with proper settings
        recreate_qdrant_collection()
    
    # Delete SQLite database
    delete_sqlite_database()
    
    # Clear logs
    clear_logs()
    
    logger.info("Cleanup completed")

if __name__ == '__main__':
    force = '--force' in sys.argv
    main(force) 