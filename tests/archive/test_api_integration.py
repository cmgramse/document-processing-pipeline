"""
Tests for Jina and Qdrant API integrations.

This module contains tests to verify:
1. API client initialization
2. Rate limiting
3. Batch processing
4. Error handling
5. Vector validation
6. Upload verification
7. Metrics tracking
"""

import pytest
import os
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import requests
from typing import List

from src.api.jina import JinaClient
from src.api.qdrant import QdrantClient

# Test data
TEST_TEXTS = [
    "This is a test document",
    "Another test document",
    "Third test document for good measure"
]

# Create test vectors using numpy for proper float handling
TEST_VECTORS = [
    np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(),
    np.array([4.0, 5.0, 6.0], dtype=np.float32).tobytes(),
    np.array([7.0, 8.0, 9.0], dtype=np.float32).tobytes()
]

@pytest.fixture
def mock_env_vars():
    """Set up test environment variables."""
    os.environ['JINA_API_KEY'] = 'test_jina_key'
    os.environ['QDRANT_API_KEY'] = 'test_qdrant_key'
    os.environ['QDRANT_URL'] = 'http://test.qdrant.url'
    yield
    del os.environ['JINA_API_KEY']
    del os.environ['QDRANT_API_KEY']
    del os.environ['QDRANT_URL']

@pytest.fixture
def jina_client(mock_env_vars):
    """Create Jina client instance."""
    return JinaClient()

@pytest.fixture
def qdrant_client(mock_env_vars):
    """Create Qdrant client instance."""
    return QdrantClient()

class TestJinaAPI:
    """Test suite for Jina AI API integration."""
    
    def test_client_initialization(self, jina_client):
        """Test client initialization with environment variables."""
        assert jina_client.api_key == 'test_jina_key'
        assert 'Authorization' in jina_client.headers
        assert jina_client.headers['Authorization'] == 'Bearer test_jina_key'
        
    def test_rate_limiting(self, jina_client):
        """Test rate limiting functionality."""
        # Set up initial state
        jina_client.embedding_calls = jina_client.LIMITS['embeddings']['rpm']
        jina_client.last_reset = datetime.now() - timedelta(seconds=30)
        
        # Verify rate limit is enforced
        assert not jina_client._check_rate_limit()
        
        # Wait for reset
        jina_client.last_reset = datetime.now() - timedelta(minutes=1)
        assert jina_client._check_rate_limit()
        assert jina_client.embedding_calls == 0
        
    @patch('requests.post')
    def test_batch_processing(self, mock_post, jina_client):
        """Test batch processing of texts."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [np.array([1.0, 2.0, 3.0], dtype=np.float32).tolist() 
                    for _ in range(len(TEST_TEXTS))]
        }
        mock_post.return_value = mock_response
        
        # Generate embeddings
        embeddings = jina_client.generate_embeddings(TEST_TEXTS)
        
        # Verify batch processing
        assert len(embeddings) == len(TEST_TEXTS)
        assert mock_post.call_count == (len(TEST_TEXTS) + jina_client.MAX_BATCH_SIZE - 1) // jina_client.MAX_BATCH_SIZE
        
    @patch('requests.post')
    def test_error_handling(self, mock_post, jina_client):
        """Test error handling and retries."""
        # Mock API error
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        # Verify error handling
        with pytest.raises(requests.exceptions.RequestException):
            jina_client.generate_embeddings(TEST_TEXTS)
            
        # Verify retry attempts
        assert mock_post.call_count == jina_client.MAX_RETRIES
        
class TestQdrantAPI:
    """Test suite for Qdrant API integration."""
    
    def test_client_initialization(self, qdrant_client):
        """Test client initialization with environment variables."""
        assert qdrant_client.api_key == 'test_qdrant_key'
        assert qdrant_client.url == 'http://test.qdrant.url'
        assert 'api-key' in qdrant_client.headers
        
    def test_vector_validation(self, qdrant_client):
        """Test vector validation."""
        # Valid vectors
        assert qdrant_client._validate_vectors(TEST_VECTORS)
        
        # Invalid vectors (different dimensions)
        invalid_vectors = [
            np.array([1.0, 2.0], dtype=np.float32).tobytes(),
            np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
        ]
        assert not qdrant_client._validate_vectors(invalid_vectors)
        
    @patch('requests.post')
    def test_batch_upload(self, mock_post, qdrant_client):
        """Test batch upload of vectors."""
        # Mock successful API responses
        mock_upload_response = MagicMock()
        mock_upload_response.json.return_value = {
            'result': {'ids': ['1', '2', '3']}
        }
        
        mock_verify_response = MagicMock()
        mock_verify_response.json.return_value = {
            'result': {'points': [{'id': '1'}, {'id': '2'}, {'id': '3'}]}
        }
        
        mock_post.side_effect = [mock_upload_response, mock_verify_response]
        
        # Upload vectors
        point_ids = qdrant_client.upload_vectors(TEST_VECTORS)
        
        # Verify upload
        assert len(point_ids) == len(TEST_VECTORS)
        assert qdrant_client.total_vectors_uploaded == len(TEST_VECTORS)
        assert qdrant_client.failed_uploads == 0
        
    @patch('requests.post')
    def test_upload_verification(self, mock_post, qdrant_client):
        """Test upload verification."""
        # Mock failed verification
        mock_upload_response = MagicMock()
        mock_upload_response.json.return_value = {
            'result': {'ids': ['1', '2', '3']}
        }
        
        mock_verify_response = MagicMock()
        mock_verify_response.json.return_value = {
            'result': {'points': [{'id': '1'}, {'id': '2'}]}  # Missing one point
        }
        
        mock_post.side_effect = [mock_upload_response, mock_verify_response]
        
        # Verify upload fails
        with pytest.raises(ValueError, match="Failed to verify upload"):
            qdrant_client.upload_vectors(TEST_VECTORS)
            
    def test_metrics_tracking(self, qdrant_client):
        """Test metrics tracking."""
        # Initial metrics
        metrics = qdrant_client.get_metrics()
        assert metrics['total_vectors_uploaded'] == 0
        assert metrics['failed_uploads'] == 0
        assert metrics['last_upload_time'] is None
        
        # Simulate failed upload
        qdrant_client.failed_uploads += 1
        qdrant_client.last_upload_time = datetime.now()
        
        # Check updated metrics
        metrics = qdrant_client.get_metrics()
        assert metrics['failed_uploads'] == 1
        assert metrics['last_upload_time'] is not None 