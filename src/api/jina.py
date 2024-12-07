import logging
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
import os

def segment_text(text: str, api_key: str) -> Dict[str, Any]:
    """Segment text using Jina AI's Segmenter API"""
    api_logger = logging.getLogger('api_calls')
    url = "https://segment.jina.ai/"
    
    # Log request details
    request_id = hashlib.md5(f"{datetime.now()}-segment".encode()).hexdigest()[:8]
    api_logger.info(f"[{request_id}] Segmenter API Request - Content length: {len(text)} chars")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "content": text,
        "tokenizer": "o200k_base",
        "return_tokens": True,
        "return_chunks": True,
        "max_chunk_length": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Log response details
        api_logger.info(
            f"[{request_id}] Segmenter API Response - "
            f"Chunks: {len(result.get('chunks', []))} | "
            f"Tokens: {result.get('num_tokens', 0)}"
        )
        return result
        
    except Exception as e:
        api_logger.error(f"[{request_id}] Segmenter API Error: {str(e)}")
        raise

def get_embeddings(texts: List[str], api_key: str, batch_size: int = 100) -> List[List[float]]:
    """Get embeddings using Jina AI's Embeddings API with batching"""
    api_logger = logging.getLogger('api_calls')
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        request_id = hashlib.md5(f"{datetime.now()}-embed-{i}".encode()).hexdigest()[:8]
        
        api_logger.info(
            f"[{request_id}] Embeddings API Request - "
            f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} | "
            f"Size: {len(batch)} texts"
        )
        
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "jina-embeddings-v3",
            "input": batch,
            "task": "retrieval.passage",
            "dimensions": 1024,
            "late_chunking": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)
            
            api_logger.info(
                f"[{request_id}] Embeddings API Response - "
                f"Successfully embedded {len(batch_embeddings)} texts"
            )
            
        except Exception as e:
            api_logger.error(f"[{request_id}] Embeddings API Error: {str(e)}")
            raise
    
    return all_embeddings 