import time
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class JinaClient:
    """Client for Jina AI APIs with rate limiting and retry logic."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting settings
        self.embedding_calls = 0
        self.rerank_calls = 0
        self.reader_calls = 0
        self.last_reset = datetime.now()
        
        # Jina API rate limits
        self.LIMITS = {
            'embeddings': {'rpm': 500, 'tpm': 1_000_000},  # Regular key limits
            'rerank': {'rpm': 500, 'tpm': 1_000_000},
            'reader': {'rpm': 200},
            'search': {'rpm': 40},
            'ground': {'rpm': 10}
        }
    
    def _check_rate_limit(self, api_type: str) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Reset counters if minute has passed
        if now - self.last_reset > timedelta(minutes=1):
            self.embedding_calls = 0
            self.rerank_calls = 0
            self.reader_calls = 0
            self.last_reset = now
        
        # Check limits
        if api_type == 'embeddings' and self.embedding_calls >= self.LIMITS['embeddings']['rpm']:
            return False
        elif api_type == 'rerank' and self.rerank_calls >= self.LIMITS['rerank']['rpm']:
            return False
        elif api_type == 'reader' and self.reader_calls >= self.LIMITS['reader']['rpm']:
            return False
            
        return True
    
    def _increment_counter(self, api_type: str) -> None:
        """Increment API call counter."""
        if api_type == 'embeddings':
            self.embedding_calls += 1
        elif api_type == 'rerank':
            self.rerank_calls += 1
        elif api_type == 'reader':
            self.reader_calls += 1
    
    @retry(stop=stop_after_attempt(3), 
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with rate limiting and retries."""
        if not self._check_rate_limit('embeddings'):
            wait_time = 60 - (datetime.now() - self.last_reset).seconds
            logger.warning(f"Rate limit reached for embeddings API. Waiting {wait_time}s")
            time.sleep(wait_time)
        
        url = "https://api.jina.ai/v1/embeddings"
        payload = {
            "model": "jina-embeddings-v3",
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            self._increment_counter('embeddings')
            
            return [data["embedding"] for data in response.json()["data"]]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Rerank documents with rate limiting and retries."""
        if not self._check_rate_limit('rerank'):
            wait_time = 60 - (datetime.now() - self.last_reset).seconds
            logger.warning(f"Rate limit reached for rerank API. Waiting {wait_time}s")
            time.sleep(wait_time)
        
        url = "https://api.jina.ai/v1/rerank"
        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            self._increment_counter('rerank')
            
            return response.json()["results"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reranking documents: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def read_url(self, url: str) -> Dict[str, Any]:
        """Read URL content with rate limiting and retries."""
        if not self._check_rate_limit('reader'):
            wait_time = 60 - (datetime.now() - self.last_reset).seconds
            logger.warning(f"Rate limit reached for reader API. Waiting {wait_time}s")
            time.sleep(wait_time)
        
        reader_url = "https://r.jina.ai/"
        headers = {
            **self.headers,
            "X-With-Links-Summary": "true",
            "X-With-Images-Summary": "true"
        }
        
        payload = {"url": url}
        
        try:
            response = requests.post(reader_url, headers=headers, json=payload)
            response.raise_for_status()
            self._increment_counter('reader')
            
            return response.json()["data"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error reading URL: {str(e)}")
            raise 