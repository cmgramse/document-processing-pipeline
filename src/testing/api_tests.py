import logging
import hashlib
from datetime import datetime
import os
from typing import Dict, Any
import requests
from pathlib import Path

from ..api.jina import segment_text, get_embeddings
from ..api.qdrant import validate_qdrant_connection

def test_document_loading(docs_path: str, glob_pattern: str) -> None:
    """Test document loading functionality"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-loading".encode()).hexdigest()[:8]
    
    try:
        # Test direct file existence
        full_path = Path(docs_path)
        api_logger.info(f"[{request_id}] Testing path: {full_path.absolute()}")
        api_logger.info(f"[{request_id}] Path exists: {full_path.exists()}")
        api_logger.info(f"[{request_id}] Files in directory: {list(full_path.glob('*'))}")
        
        # Test DirectoryLoader
        loader = DirectoryLoader(docs_path, glob=glob_pattern)
        docs = loader.load()
        api_logger.info(
            f"[{request_id}] DirectoryLoader found {len(docs)} documents "
            f"using pattern: {glob_pattern}"
        )
        
        # Print details of each found document
        for doc in docs:
            api_logger.info(
                f"[{request_id}] Found document: {doc.metadata['source']}, "
                f"Size: {len(doc.page_content)} chars"
            )
            
    except Exception as e:
        api_logger.error(f"[{request_id}] Document loading test failed: {str(e)}")
        raise

def test_jina_apis(test_text: str = "This is a test document. Let's see how it gets processed."):
    """Test Jina API functionality"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-apis".encode()).hexdigest()[:8]
    
    try:
        # Test Segmenter API
        api_logger.info(f"[{request_id}] Testing Jina Segmenter API...")
        segments = segment_text(test_text, os.environ["JINA_API_KEY"])
        api_logger.info(
            f"[{request_id}] Segmenter test successful: "
            f"Generated {len(segments.get('chunks', []))} chunks"
        )
        
        # Test Embeddings API
        if segments.get("chunks"):
            api_logger.info(f"[{request_id}] Testing Jina Embeddings API...")
            embeddings = get_embeddings(segments["chunks"], os.environ["JINA_API_KEY"])
            api_logger.info(
                f"[{request_id}] Embeddings test successful: "
                f"Generated {len(embeddings)} embeddings"
            )
            
        return True
        
    except Exception as e:
        api_logger.error(f"[{request_id}] API test failed: {str(e)}")
        raise

def test_qdrant_connection():
    """Test Qdrant connection and basic operations"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-qdrant".encode()).hexdigest()[:8]
    
    try:
        # Test basic connection
        api_logger.info(f"[{request_id}] Testing Qdrant connection...")
        qdrant_url = os.environ["QDRANT_URL"]
        response = requests.get(
            f"{qdrant_url}/collections",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        response.raise_for_status()
        
        # Test collection existence
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        response = requests.get(
            f"{qdrant_url}/collections/{collection_name}",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        
        if response.status_code == 200:
            collection_info = response.json()
            api_logger.info(
                f"[{request_id}] Collection exists: {collection_name}\n"
                f"Vector size: {collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size')}\n"
                f"Points count: {collection_info.get('points_count')}"
            )
        else:
            api_logger.warning(f"[{request_id}] Collection {collection_name} does not exist")
        
        return True
        
    except Exception as e:
        api_logger.error(f"[{request_id}] Qdrant test failed: {str(e)}")
        raise 