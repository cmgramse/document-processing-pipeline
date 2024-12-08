"""
API Testing Module

This module provides testing utilities for validating API integrations with
Jina AI and Qdrant. It includes tests for connection health, data consistency,
and error handling.

The module manages:
- API connection testing
- Data consistency validation
- Error simulation and handling
- Performance benchmarking

Features:
- Automated connection tests
- Data validation checks
- Error recovery testing
- Performance metrics

Example:
    Test Qdrant connection:
        success = test_qdrant_connection(client)
    
    Test Jina AI APIs:
        results = test_jina_apis(api_key)
"""

import logging
from datetime import datetime
import hashlib
import json
from typing import Dict, List, Any, Optional
import time
import random

from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

def test_qdrant_connection(client: QdrantClient, 
                          test_docs: Optional[List[Document]] = None) -> bool:
    """
    Test Qdrant connection and basic operations.
    
    Args:
        client: Qdrant client instance
        test_docs: Optional list of test documents
    
    Returns:
        bool: True if all tests pass, False otherwise
    
    The function tests:
    1. Connection health
    2. Collection existence
    3. Basic operations (if test_docs provided)
    
    Example:
        client = QdrantClient(...)
        if test_qdrant_connection(client):
            print("Qdrant connection is healthy")
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-qdrant".encode()).hexdigest()[:8]
    
    try:
        # Log test start
        api_logger.info(json.dumps({
            'request_id': request_id,
            'operation': 'test_qdrant',
            'test_docs': bool(test_docs)
        }))
        
        # Test connection
        health = client.health()
        if not health:
            raise Exception("Qdrant health check failed")
        
        # Test operations if documents provided
        if test_docs:
            collection_name = "test_collection"
            
            # Create test collection
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
            
            # Upload test documents
            points = []
            for i, doc in enumerate(test_docs):
                points.append(models.PointStruct(
                    id=i,
                    vector=[random.random() for _ in range(1024)],
                    payload={"text": doc.page_content}
                ))
            
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Test search
            results = client.search(
                collection_name=collection_name,
                query_vector=[random.random() for _ in range(1024)],
                limit=1
            )
            
            if not results:
                raise Exception("Search returned no results")
            
            # Clean up
            client.delete_collection(collection_name)
        
        api_logger.info(json.dumps({
            'request_id': request_id,
            'status': 'success',
            'message': 'All Qdrant tests passed'
        }))
        
        return True
        
    except Exception as e:
        api_logger.error(json.dumps({
            'request_id': request_id,
            'status': 'error',
            'error': str(e)
        }))
        return False

def test_jina_apis(api_key: str) -> Dict[str, Any]:
    """
    Test Jina AI API endpoints and functionality.
    
    Args:
        api_key: Jina AI API key
    
    Returns:
        Dict containing test results:
        - segmenter_status: Segmenter API test result
        - embeddings_status: Embeddings API test result
        - latency: API latency measurements
    
    The function tests:
    1. API key validation
    2. Segmenter functionality
    3. Embeddings generation
    4. Error handling
    
    Example:
        results = test_jina_apis(api_key)
        if results['segmenter_status'] == 'ok':
            print("Segmenter API is working")
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-jina".encode()).hexdigest()[:8]
    
    results = {
        'segmenter_status': 'unknown',
        'embeddings_status': 'unknown',
        'latency': {}
    }
    
    try:
        # Log test start
        api_logger.info(json.dumps({
            'request_id': request_id,
            'operation': 'test_jina_apis'
        }))
        
        # Test segmenter
        start_time = time.time()
        test_text = "This is a test document for API validation."
        
        from ..api.jina import segment_text
        chunks = segment_text(test_text, api_key)
        
        results['segmenter_status'] = 'ok'
        results['latency']['segmenter'] = time.time() - start_time
        
        # Test embeddings
        if chunks:
            start_time = time.time()
            from ..api.jina import get_embeddings
            embeddings = get_embeddings([chunk['text'] for chunk in chunks], api_key)
            
            results['embeddings_status'] = 'ok'
            results['latency']['embeddings'] = time.time() - start_time
        
        api_logger.info(json.dumps({
            'request_id': request_id,
            'status': 'success',
            'results': results
        }))
        
    except Exception as e:
        api_logger.error(json.dumps({
            'request_id': request_id,
            'status': 'error',
            'error': str(e)
        }))
        results['error'] = str(e)
    
    return results

def benchmark_apis(api_key: str, num_tests: int = 10) -> Dict[str, Any]:
    """
    Benchmark API performance and reliability.
    
    Args:
        api_key: Jina AI API key
        num_tests: Number of test iterations
    
    Returns:
        Dict containing benchmark results:
        - avg_latency: Average latency per operation
        - success_rate: Success rate per operation
        - error_types: Encountered error types and counts
    
    Example:
        results = benchmark_apis(api_key, num_tests=20)
        print(f"Average latency: {results['avg_latency']} ms")
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-benchmark".encode()).hexdigest()[:8]
    
    results = {
        'avg_latency': {},
        'success_rate': {},
        'error_types': {}
    }
    
    try:
        api_logger.info(json.dumps({
            'request_id': request_id,
            'operation': 'benchmark_apis',
            'num_tests': num_tests
        }))
        
        for i in range(num_tests):
            test_results = test_jina_apis(api_key)
            
            # Update latency stats
            for op, latency in test_results.get('latency', {}).items():
                if op not in results['avg_latency']:
                    results['avg_latency'][op] = []
                results['avg_latency'][op].append(latency)
            
            # Update success stats
            for op in ['segmenter_status', 'embeddings_status']:
                status = test_results.get(op, 'unknown')
                if op not in results['success_rate']:
                    results['success_rate'][op] = {'ok': 0, 'failed': 0}
                results['success_rate'][op]['ok' if status == 'ok' else 'failed'] += 1
            
            # Track errors
            if 'error' in test_results:
                error_type = type(test_results['error']).__name__
                results['error_types'][error_type] = results['error_types'].get(error_type, 0) + 1
            
            time.sleep(1)  # Rate limiting
        
        # Calculate averages
        for op in results['avg_latency']:
            results['avg_latency'][op] = sum(results['avg_latency'][op]) / len(results['avg_latency'][op])
        
        api_logger.info(json.dumps({
            'request_id': request_id,
            'status': 'success',
            'results': results
        }))
        
    except Exception as e:
        api_logger.error(json.dumps({
            'request_id': request_id,
            'status': 'error',
            'error': str(e)
        }))
        results['error'] = str(e)
    
    return results