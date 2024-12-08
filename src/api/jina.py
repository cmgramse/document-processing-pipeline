"""
Jina AI Integration Module

This module provides a unified interface for all Jina AI services:
- Text segmentation using o200k_base segmenter
- Embedding generation using jina-embeddings-v3
- Batch processing and rate limiting
- Error handling and retries

Required Environment Variables:
    JINA_API_KEY: API key for Jina AI services
    JINA_EMBEDDING_MODEL: Embedding model name (default: jina-embeddings-v3)
    JINA_SEGMENTER_MODEL: Segmenter model name (default: o200k_base)
"""

import os
import logging
import hashlib
from datetime import datetime
import json
import time
import requests
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from src.monitoring.metrics import log_api_call

class JinaAPI:
    """Unified interface for Jina AI services."""
    
    def __init__(self):
        """Initialize Jina API client with environment variables."""
        self.api_key = os.environ["JINA_API_KEY"]
        self.embedding_model = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
        self.segmenter_model = os.environ.get("JINA_SEGMENTER_MODEL", "o200k_base")
        self.api_logger = logging.getLogger('api_calls')

    def _log_operation(self, operation: str, details: Dict[str, Any], success: bool = True) -> str:
        """Log API operations with request ID."""
        status = "SUCCESS" if success else "FAILED"
        request_id = hashlib.md5(f"{datetime.now()}-{operation}".encode()).hexdigest()[:8]
        
        self.api_logger.info(
            f"[{request_id}] Jina {operation} {status} - "
            f"Details: {json.dumps(details)}"
        )
        return request_id

    def segment_text(self, text: str, retry_count: int = 3) -> List[Document]:
        """
        Segment text into semantic chunks using Jina AI's o200k_base segmenter.
        
        Args:
            text: Input text to segment
            retry_count: Number of retries on failure
            
        Returns:
            List of Document objects containing chunks
        """
        start_time = time.time()
        request_id = self._log_operation("Segmentation", {
            "text_length": len(text),
            "model": self.segmenter_model
        })
        
        url = "https://api.jina.ai/v1/segmenter"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model": self.segmenter_model,
            "task": "segmentation.document",
            "options": {
                "overlap": True,
                "min_length": 100,
                "max_length": 2048
            }
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if not isinstance(result, dict) or 'chunks' not in result:
                    raise ValueError("Invalid response format from Jina API")
                
                chunks = []
                for i, chunk in enumerate(result['chunks']):
                    chunks.append(Document(
                        page_content=chunk['text'],
                        metadata={
                            'chunk_number': i,
                            'total_chunks': len(result['chunks']),
                            'start': chunk.get('start', 0),
                            'end': chunk.get('end', len(chunk['text'])),
                            'model': self.segmenter_model
                        }
                    ))
                
                log_api_call(
                    operation="segment_text",
                    start_time=start_time,
                    success=True,
                    details={
                        "input_length": len(text),
                        "chunks_created": len(chunks)
                    }
                )
                
                self._log_operation("Segmentation Success", {
                    "request_id": request_id,
                    "chunks_created": len(chunks)
                })
                
                return chunks
                
            except Exception as e:
                if attempt == retry_count - 1:
                    self._log_operation("Segmentation Failed", {
                        "request_id": request_id,
                        "error": str(e),
                        "attempt": attempt + 1
                    }, success=False)
                    raise
                
                wait_time = min(2 ** attempt, 60)
                self.api_logger.warning(
                    f"[{request_id}] Segmentation failed, retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

    def get_embeddings(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings for texts using Jina AI's embedding model.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors (1024 dimensions each)
        """
        start_time = time.time()
        request_id = self._log_operation("Embeddings", {
            "total_texts": len(texts),
            "batch_size": batch_size,
            "model": self.embedding_model
        })
        
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_id = f"{request_id}-batch-{i//batch_size + 1}"
            
            self.api_logger.info(
                f"[{batch_id}] Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
            )
            
            payload = {
                "model": self.embedding_model,
                "input": [str(text).strip() for text in batch],
                "task": "retrieval.passage",
                "late_chunking": True,
                "dimensions": 1024,
                "embedding_type": "float"
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if not isinstance(result, dict) or 'data' not in result:
                    raise ValueError("Invalid response format from Jina API")
                
                batch_embeddings = [item['embedding'] for item in result['data']]
                
                # Verify embedding dimensions
                if batch_embeddings and len(batch_embeddings[0]) != 1024:
                    raise ValueError(f"Unexpected embedding dimensions: {len(batch_embeddings[0])}")
                
                all_embeddings.extend(batch_embeddings)
                
                self.api_logger.info(
                    f"[{batch_id}] Successfully embedded {len(batch)} texts"
                )
                
            except Exception as e:
                self._log_operation("Embeddings Failed", {
                    "request_id": request_id,
                    "batch": i//batch_size + 1,
                    "error": str(e)
                }, success=False)
                raise
        
        log_api_call(
            operation="get_embeddings",
            start_time=start_time,
            success=True,
            details={
                "texts_count": len(texts),
                "embeddings_count": len(all_embeddings)
            }
        )
        
        self._log_operation("Embeddings Success", {
            "request_id": request_id,
            "total_embeddings": len(all_embeddings)
        })
        
        return all_embeddings

    def process_document(self, text: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a document through the complete pipeline: segmentation and embedding.
        
        Args:
            text: Document text to process
            source: Optional source identifier for the document
            
        Returns:
            List of dictionaries containing chunks and their embeddings
        """
        start_time = time.time()
        request_id = self._log_operation("Document Processing", {
            "text_length": len(text),
            "source": source
        })
        
        try:
            # Step 1: Segment text
            chunks = self.segment_text(text)
            
            # Step 2: Generate embeddings
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self.get_embeddings(chunk_texts)
            
            # Step 3: Combine results
            processed_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                metadata = chunk.metadata.copy()
                if source:
                    metadata['source'] = source
                
                processed_chunks.append({
                    'content': chunk.page_content,
                    'embedding': embedding,
                    'metadata': metadata
                })
            
            log_api_call(
                operation="process_document",
                start_time=start_time,
                success=True,
                details={
                    "source": source,
                    "chunks_processed": len(processed_chunks)
                }
            )
            
            self._log_operation("Document Processing Success", {
                "request_id": request_id,
                "chunks_processed": len(processed_chunks)
            })
            
            return processed_chunks
            
        except Exception as e:
            log_api_call(
                operation="process_document",
                start_time=start_time,
                success=False,
                details={
                    "source": source,
                    "error": str(e)
                }
            )
            self._log_operation("Document Processing Failed", {
                "request_id": request_id,
                "error": str(e)
            }, success=False)
            raise

# Create a singleton instance for easy access
jina_client = JinaAPI()

# Convenience functions that use the singleton instance
def segment_text(text: str) -> List[Document]:
    """Convenience function for text segmentation."""
    return jina_client.segment_text(text)

def get_embeddings(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """Convenience function for embedding generation."""
    return jina_client.get_embeddings(texts, batch_size)

def process_document(text: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function for complete document processing."""
    return jina_client.process_document(text, source)