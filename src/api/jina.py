"""
Jina AI API integration.
"""

import os
import logging
import json
import requests
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for a text using Jina AI API.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of floats representing the embedding, or None if failed
    """
    try:
        # Prepare request
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
            "input": [text]
        }
        
        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if not result.get("data") or not result["data"][0].get("embedding"):
            raise ValueError("No embedding in response")
            
        return result["data"][0]["embedding"]
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None

def get_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to generate embeddings for
        
    Returns:
        List of embeddings (each is a list of floats), or None for failed texts
    """
    try:
        # Prepare request
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "model": os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
            "input": texts
        }
        
        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if not result.get("data"):
            raise ValueError("No embeddings in response")
            
        return [item.get("embedding") for item in result["data"]]
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return [None] * len(texts)

def segment_text(text: str) -> List[str]:
    """
    Segment text into chunks using Jina AI Segmenter API.
    
    Args:
        text: Text to segment
        
    Returns:
        List of text segments
    """
    try:
        # Prepare request
        url = "https://segment.jina.ai/"
        headers = {
            "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "content": text,
            "return_chunks": True,
            "max_chunk_length": 1000  # Adjust as needed
        }
        
        # Make request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if not result.get("chunks"):
            raise ValueError("No chunks in response")
            
        return result["chunks"]
        
    except Exception as e:
        logger.error(f"Failed to segment text: {e}")
        return [text]  # Return original text as single chunk on failure