"""
Qdrant vector database integration.
"""

import logging
import hashlib
from datetime import datetime
import os
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from functools import wraps

# Optional dependencies that enhance functionality but aren't strictly required
numpy_available = False
langchain_available = False
retry_available = False

try:
    import numpy as np
    numpy_available = True
except ImportError:
    np = None
    logging.warning("numpy is not installed. Vector validation will be limited.")

try:
    from langchain.schema import Document
    from langchain_community.embeddings import JinaEmbeddings
    from langchain_community.vectorstores import Qdrant
    langchain_available = True
except ImportError:
    Document = Any  # type: ignore
    logging.warning("langchain packages are not installed. Some features will be limited.")

try:
    import backoff
    from tenacity import retry, stop_after_attempt, wait_exponential
    retry_available = True
except ImportError:
    logging.warning("backoff and tenacity are not installed. Retry functionality will be disabled.")
    
    # Dummy decorator when retry packages aren't available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def stop_after_attempt(*args, **kwargs):
        pass
    
    def wait_exponential(*args, **kwargs):
        pass

from ..config.settings import settings
from ..monitoring.metrics import log_api_call

logger = logging.getLogger(__name__)

# Constants for retry and backoff
MAX_RETRIES = 3
INITIAL_WAIT = 1  # seconds
MAX_WAIT = 30  # seconds
VECTOR_DIMENSION = 1024

class QdrantError(Exception):
    """Base exception for Qdrant operations."""
    pass

class QdrantConnectionError(QdrantError):
    """Raised when connection to Qdrant fails."""
    pass

class QdrantValidationError(Exception):
    """Raised when vector or metadata validation fails."""
    pass

def log_qdrant_operation(operation: str, details: Dict[str, Any], success: bool = True) -> str:
    """Log Qdrant operations with request ID."""
    status = "SUCCESS" if success else "FAILED"
    request_id = hashlib.md5(f"{datetime.now()}-{operation}".encode()).hexdigest()[:8]
    
    logger.info(
        f"[{request_id}] Qdrant {operation} {status} - "
        f"Details: {json.dumps(details)}"
    )
    
    # Log to metrics system
    log_api_call(
        operation=f"qdrant_{operation.lower()}",
        start_time=datetime.now().timestamp(),
        success=success,
        details=details
    )
    
    return request_id

def validate_vector_data(vectors: List[List[float]], operation: str) -> None:
    """
    Validate vector data before operations.
    Does not interrupt existing flow if validation fails.
    Basic validation works without numpy, but advanced validation requires it.
    
    Args:
        vectors: List of vectors to validate
        operation: Name of the operation being performed
    """
    try:
        if not vectors:
            logging.warning(f"{operation}: Empty vector list")
            return
            
        # Check vector dimensions
        for i, vector in enumerate(vectors):
            if len(vector) != VECTOR_DIMENSION:
                logging.warning(
                    f"{operation}: Vector {i} has incorrect dimension "
                    f"{len(vector)} (expected {VECTOR_DIMENSION})"
                )
            
            if numpy_available:
                # Advanced validation with numpy
                if not all(np.isfinite(x) for x in vector):
                    logging.warning(f"{operation}: Vector {i} contains NaN or Inf values")
                
                magnitude = np.linalg.norm(vector)
                if magnitude < 1e-6:
                    logging.warning(f"{operation}: Vector {i} has very small magnitude: {magnitude}")
            else:
                # Basic validation without numpy
                if not all(isinstance(x, (int, float)) for x in vector):
                    logging.warning(f"{operation}: Vector {i} contains non-numeric values")
                
    except Exception as e:
        logging.error(f"Vector validation error in {operation}: {str(e)}")

def with_retries(func):
    """
    Decorator that adds retry logic to operations while preserving existing logging.
    Falls back to no retries if retry packages aren't available.
    """
    if not retry_available:
        return func
        
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT),
        reraise=True
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def validate_metadata(metadata: Dict[str, Any], operation: str) -> None:
    """
    Validate metadata before operations.
    
    Args:
        metadata: Metadata dictionary to validate
        operation: Name of the operation being performed
        
    Raises:
        QdrantValidationError: If validation fails
    """
    required_fields = {'filename', 'chunk_number'}
    missing_fields = required_fields - set(metadata.keys())
    if missing_fields:
        raise QdrantValidationError(
            f"{operation}: Missing required metadata fields: {missing_fields}"
        )
    
    # Validate field types
    if not isinstance(metadata.get('filename', ''), str):
        raise QdrantValidationError(f"{operation}: 'filename' must be a string")
    if not isinstance(metadata.get('chunk_number', 0), int):
        raise QdrantValidationError(f"{operation}: 'chunk_number' must be an integer")
    
    # Validate field lengths
    if len(metadata.get('filename', '')) > 1000:  # Arbitrary limit
        raise QdrantValidationError(f"{operation}: 'filename' field too long")

@with_retries
def validate_qdrant_connection() -> bool:
    """Validate connection to Qdrant server with retry logic."""
    try:
        qdrant_url = os.environ["QDRANT_URL"]
        response = requests.get(
            f"{qdrant_url}/collections",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        response.raise_for_status()
        return True
        
    except Exception as e:
        raise QdrantConnectionError(f"Failed to connect to Qdrant: {str(e)}")

@with_retries
def delete_vectors_by_filter(filter_dict: Dict[str, Any]) -> bool:
    """Delete vectors with retry logic using HTTP API."""
    logger = logging.getLogger('pipeline')
    try:
        qdrant_url = os.environ["QDRANT_URL"]
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Log operation start
        request_id = log_qdrant_operation("Delete by Filter", {
            'filter': filter_dict,
            'collection': collection_name
        })
        
        # Prepare payload based on input type
        if "points" in filter_dict:
            # Direct points deletion
            filter_payload = {
                "points": filter_dict["points"]
            }
            logger.info(f"Deleting {len(filter_dict['points'])} specific points")
        elif "must" in filter_dict:
            # Filter-based deletion
            filter_payload = {
                "filter": filter_dict
            }
            logger.info("Deleting points based on filter criteria")
        else:
            # Use as is
            filter_payload = filter_dict
            logger.info("Using custom deletion payload")
        
        logger.debug(f"Delete payload: {json.dumps(filter_payload)}")
        
        response = requests.post(
            f"{qdrant_url}/collections/{collection_name}/points/delete",
            headers=headers,
            json=filter_payload
        )
        response.raise_for_status()
        
        # Log success
        log_qdrant_operation("Delete Success", {
            'request_id': request_id,
            'filter': filter_dict,
            'response': response.json() if response.text else {}
        })
        
        return True
        
    except Exception as e:
        error_details = {
            'error': str(e),
            'filter': filter_dict,
            'response': response.text if 'response' in locals() else None
        }
        log_qdrant_operation("Delete Failed", error_details, success=False)
        raise QdrantError(f"Failed to delete vectors: {str(e)}")

@with_retries
def upsert_embeddings(chunks: List[Document], embeddings: List[List[float]]) -> bool:
    """
    Upsert document chunks and their embeddings to Qdrant using HTTP API.
    Enhanced with validation and retry logic while preserving existing functionality.
    """
    try:
        # Add validation without interrupting flow
        validate_vector_data(embeddings, "upsert_embeddings")
        
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        qdrant_url = os.environ["QDRANT_URL"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Prepare points for upserting
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = hashlib.md5(
                f"{chunk.metadata['source']}-{chunk.page_content[:100]}"
                .encode()
            ).hexdigest()
            
            points.append({
                'id': point_id,
                'vector': embedding,
                'payload': {
                    'text': chunk.page_content,
                    'source': chunk.metadata['source'],
                    'chunk_id': chunk.metadata.get('chunk_id', ''),
                    'updated_at': datetime.now().isoformat()
                }
            })
        
        # Log operation start
        request_id = log_qdrant_operation("Upsert", {
            'num_chunks': len(chunks),
            'collection': collection_name,
            'vector_dim': len(embeddings[0]) if embeddings else 0
        })
        
        # Perform upsert in batches with enhanced progress tracking
        batch_size = 100
        total_batches = (len(points) - 1) // batch_size + 1
        
        @with_retries
        def do_upsert(batch):
            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}/points",
                headers=headers,
                json={"points": batch}
            )
            response.raise_for_status()
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logging.info(
                f"Upserting batch {batch_num}/{total_batches} "
                f"({len(batch)} points)"
            )
            
            do_upsert(batch)
        
        # Log success
        log_qdrant_operation("Upsert Success", {
            'request_id': request_id,
            'points_upserted': len(points)
        })
        
        return True
        
    except Exception as e:
        # Log failure (preserving existing error handling)
        error_details = {
            'error': str(e),
            'num_chunks': len(chunks)
        }
        log_qdrant_operation("Upsert Failed", error_details, success=False)
        logging.error(f"Failed to upsert embeddings: {str(e)}")
        return False

class QdrantClient:
    """Client for interacting with Qdrant vector database."""
    
    def __init__(self, document_manager=None):
        """Initialize Qdrant client with environment variables."""
        self.url = os.environ["QDRANT_URL"]
        self.api_key = os.environ["QDRANT_API_KEY"]
        self.collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.document_manager = document_manager
        self._validate_connection()
    
    def _validate_connection(self):
        """Validate connection on initialization."""
        if not validate_qdrant_connection():
            raise QdrantConnectionError("Failed to establish connection")

    def _update_status(self, filename: str, status: str = 'completed') -> None:
        """Update document status if document manager is available."""
        if self.document_manager:
            self.document_manager.update_qdrant_status(filename, status)

    @with_retries
    def upload_vectors(self, vectors: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """
        Upload vectors to Qdrant with retry logic using HTTP API.
        
        Args:
            vectors: List of vectors to upload
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of Qdrant point IDs
        """
        if not vectors or not metadata_list:
            raise QdrantValidationError("Empty vectors or metadata list")
        
        if len(vectors) != len(metadata_list):
            raise QdrantValidationError("Vectors and metadata lists must have same length")
        
        # Validate vectors
        validate_vector_data(vectors, "Upload")
        
        try:
            qdrant_url = os.environ["QDRANT_URL"]
            collection_name = os.environ["QDRANT_COLLECTION_NAME"]
            headers = {
                "api-key": os.environ["QDRANT_API_KEY"],
                "Content-Type": "application/json"
            }
            
            # Generate point IDs
            point_ids = [
                hashlib.md5(f"{m['filename']}-{m['chunk_number']}".encode()).hexdigest()
                for m in metadata_list
            ]
            
            # Prepare payload
            payload = {
                "points": [
                    {
                        "id": point_id,
                        "vector": vector,
                        "payload": metadata
                    }
                    for point_id, vector, metadata in zip(point_ids, vectors, metadata_list)
                ]
            }
            
            # Log operation start
            request_id = log_qdrant_operation("Upload", {
                'points': len(vectors),
                'collection': collection_name
            })
            
            # Make request
            response = requests.put(
                f"{qdrant_url}/collections/{collection_name}/points",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Log success
            log_qdrant_operation("Upload Success", {
                'request_id': request_id,
                'points': len(vectors),
                'response': response.json() if response.text else {}
            })
            
            return point_ids
            
        except Exception as e:
            log_qdrant_operation("Upload Failed", {
                'error': str(e),
                'points': len(vectors) if vectors else 0
            }, success=False)
            raise QdrantError(f"Failed to upload vectors: {str(e)}")

    @with_retries
    def similarity_search(self, query: List[Any], k: int = 5) -> Dict[str, Any]:
        """Performs a similarity search on Qdrant with the given query vector."""
        # Add validation before existing logic
        validate_vector_data([query], "similarity_search")
        
        request_id = log_qdrant_operation("Similarity Search", {
            "query_vector": query,
            "k": k
        })
        
        try:
            response = requests.post(
                f"{self.url}/collections/{self.collection_name}/search",
                headers=self.headers,
                json={"vectors": [query], "k": k}
            )
            response.raise_for_status()
            
            results = response.json()["results"]
            log_qdrant_operation("Similarity Search Success", {
                "status_code": response.status_code,
                "num_results_found": len(results)
            })
            return results
            
        except Exception as e:
            error_details = {
                "error": str(e),
                "query": query,
                "k": k
            }
            log_qdrant_operation("Search Failed", error_details, success=False)
            logging.error(f"Failed to perform similarity search: {str(e)}")
            raise  # Let retry decorator handle the retry logic
            
    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            response = requests.delete(
                f"{self.url}/collections/{self.collection_name}",
                headers=self.headers
            )
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:  # Collection doesn't exist
                return True
            raise
            
    def create_collection(self, vector_size: int = VECTOR_DIMENSION) -> bool:
        """Create a new collection with specified vector size."""
        collection_config = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        }
        
        response = requests.put(
            f"{self.url}/collections/{self.collection_name}",
            headers=self.headers,
            json=collection_config
        )
        response.raise_for_status()
        return True

@with_retries
def get_all_vector_ids() -> List[str]:
    """Get all vector IDs from Qdrant collection."""
    try:
        qdrant_url = os.environ["QDRANT_URL"]
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Log operation start
        request_id = log_qdrant_operation("Get All Vector IDs", {
            'collection': collection_name
        })
        
        # Scroll through all vectors to get their IDs
        scroll_param = None
        all_ids = []
        
        while True:
            url = f"{qdrant_url}/collections/{collection_name}/points/scroll"
            payload = {
                "limit": 100,
                "with_payload": False,
                "with_vector": False
            }
            if scroll_param:
                payload["offset"] = scroll_param
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            points = data.get('result', {}).get('points', [])
            if not points:
                break
                
            all_ids.extend(str(point['id']) for point in points)
            scroll_param = data.get('result', {}).get('next_page_offset')
            if not scroll_param:
                break
        
        # Log success
        log_qdrant_operation("Get All Vector IDs Success", {
            'request_id': request_id,
            'count': len(all_ids)
        })
        
        return all_ids
        
    except Exception as e:
        error_details = {'error': str(e)}
        log_qdrant_operation("Get All Vector IDs Failed", error_details, success=False)
        raise QdrantError(f"Failed to get vector IDs: {str(e)}")

@with_retries
def verify_vector_deletion(vector_ids: List[str]) -> bool:
    """Verify that vectors have been deleted from Qdrant."""
    try:
        qdrant_url = os.environ["QDRANT_URL"]
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Log operation start
        request_id = log_qdrant_operation("Verify Vector Deletion", {
            'vector_ids': vector_ids,
            'collection': collection_name
        })
        
        # Check for existence of vectors
        response = requests.post(
            f"{qdrant_url}/collections/{collection_name}/points/scroll",
            headers=headers,
            json={
                "filter": {
                    "must": [
                        {"id": {"in": vector_ids}}
                    ]
                },
                "limit": len(vector_ids),
                "with_payload": False,
                "with_vector": False
            }
        )
        response.raise_for_status()
        
        remaining_vectors = response.json().get('result', {}).get('points', [])
        success = len(remaining_vectors) == 0
        
        # Log result
        log_qdrant_operation("Verify Vector Deletion Complete", {
            'request_id': request_id,
            'success': success,
            'remaining_vectors': len(remaining_vectors)
        })
        
        return success
        
    except Exception as e:
        error_details = {
            'error': str(e),
            'vector_ids': vector_ids
        }
        log_qdrant_operation("Verify Vector Deletion Failed", error_details, success=False)
        raise QdrantError(f"Failed to verify vector deletion: {str(e)}")

@with_retries
def cleanup_orphaned_vectors(all_vector_ids: List[str], failed_vector_ids: List[str], force: bool = False) -> Tuple[int, List[str]]:
    """
    Clean up orphaned vectors in Qdrant.
    
    Handles two types of orphaned vectors:
    1. Vectors whose documents have been completely deleted from SQLite
    2. Vectors whose documents failed to upload (status = 'failed')
    
    Args:
        all_vector_ids: List of all vector IDs from SQLite
        failed_vector_ids: List of vector IDs from documents that failed to upload
        force: Skip confirmation prompt if True
        
    Returns:
        Tuple of (number of vectors deleted, list of deleted vector IDs)
    """
    logger = logging.getLogger('pipeline')
    try:
        # Get all vector IDs from Qdrant
        qdrant_vector_ids = get_all_vector_ids()
        
        # Print status to console
        print(f"Found {len(qdrant_vector_ids)} vectors in Qdrant")
        print(f"Found {len(all_vector_ids)} total vectors in SQLite")
        print(f"Found {len(failed_vector_ids)} failed vectors in SQLite")
        
        # Debug: Print sample IDs to understand format
        print("\nDebug - Sample vector IDs:")
        print("Qdrant vector ID format:", qdrant_vector_ids[0] if qdrant_vector_ids else "None")
        print("SQLite vector ID format:", all_vector_ids[0] if all_vector_ids else "None")
        
        def normalize_id(id_str: str) -> str:
            """Normalize ID by removing hyphens and converting to lowercase."""
            return str(id_str).replace('-', '').strip().lower()
        
        # Convert to sets for efficient comparison
        # Normalize IDs by removing hyphens and converting to lowercase
        qdrant_ids_set = {normalize_id(id) for id in qdrant_vector_ids}
        all_ids_set = {normalize_id(id) for id in all_vector_ids}
        failed_ids_set = {normalize_id(id) for id in failed_vector_ids}
        
        # Find orphaned vectors:
        # 1. Vectors in Qdrant but not in SQLite at all (completely deleted)
        # 2. Vectors in Qdrant that belong to failed documents
        completely_deleted = qdrant_ids_set - all_ids_set
        failed_uploads = qdrant_ids_set.intersection(failed_ids_set)
        
        # Debug: Print set sizes
        print("\nDebug - Set sizes:")
        print(f"Qdrant IDs: {len(qdrant_ids_set)}")
        print(f"SQLite IDs: {len(all_ids_set)}")
        print(f"Common IDs: {len(qdrant_ids_set.intersection(all_ids_set))}")
        
        # Map normalized IDs back to original Qdrant IDs for deletion
        orphaned_normalized = completely_deleted | failed_uploads
        orphaned_ids = [id for id in qdrant_vector_ids if normalize_id(id) in orphaned_normalized]
        
        if not orphaned_ids:
            print("\nNo orphaned vectors found")
            return 0, []
        
        # Print orphaned vector details to console
        print(f"\nFound {len(orphaned_ids)} total orphaned vectors:")
        print(f"- {len(completely_deleted)} vectors from deleted documents")
        print(f"- {len(failed_uploads)} vectors from failed uploads")
        
        # Debug: Print sample orphaned IDs
        print("\nDebug - Sample orphaned vector IDs:")
        for idx, vid in enumerate(sorted(orphaned_ids)[:3], 1):
            print(f"{idx}. {vid}")
            print(f"   Normalized: {normalize_id(vid)}")
            print(f"   In SQLite: {normalize_id(vid) in all_ids_set}")
        
        if not force:
            # Ask for confirmation
            while True:
                confirm = input(f"\nDo you want to delete {len(orphaned_ids)} orphaned vectors? (yes/no): ").lower()
                if confirm in ['yes', 'y']:
                    break
                elif confirm in ['no', 'n']:
                    print("Vector cleanup cancelled by user")
                    return 0, []
                print("Please answer 'yes' or 'no'")
        
        # Confirm before deletion
        print(f"\nPreparing to delete {len(orphaned_ids)} orphaned vectors")
        
        # Delete orphaned vectors using points format
        success = delete_vectors_by_filter({
            "points": orphaned_ids  # Direct points deletion
        })
        
        if not success:
            raise QdrantError("Failed to delete orphaned vectors")
        
        # Verify deletion
        if not verify_vector_deletion(orphaned_ids):
            raise QdrantError("Failed to verify deletion of orphaned vectors")
        
        # Log success
        print(f"\nSuccessfully deleted {len(orphaned_ids)} orphaned vectors")
        
        return len(orphaned_ids), orphaned_ids
        
    except Exception as e:
        error_details = {
            'error': str(e),
            'all_vector_count': len(all_vector_ids),
            'failed_vector_count': len(failed_vector_ids),
            'qdrant_vector_count': len(qdrant_vector_ids) if 'qdrant_vector_ids' in locals() else 0
        }
        log_qdrant_operation("Cleanup Orphaned Vectors Failed", error_details, success=False)
        raise QdrantError(f"Failed to cleanup orphaned vectors: {str(e)}")

@with_retries
def upload_embeddings(chunk_ids: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
    """
    Upload embeddings to Qdrant with retry logic using HTTP API.
    
    Args:
        chunk_ids: List of chunk IDs to use as point IDs
        embeddings: List of embedding vectors
        metadata: List of metadata dictionaries for each vector
        
    Returns:
        List of Qdrant point IDs for the uploaded vectors
    """
    if len(chunk_ids) != len(embeddings) or len(embeddings) != len(metadata):
        raise ValueError("Length mismatch between IDs, embeddings, and metadata")
        
    # Validate vectors
    validate_vector_data(embeddings, "Upload")
    
    # Validate metadata
    for meta in metadata:
        validate_metadata(meta, "Upload")
    
    # Log operation start
    request_id = log_qdrant_operation("Upload", {
        'vectors_count': len(embeddings),
        'collection': os.environ["QDRANT_COLLECTION_NAME"]
    })
    
    try:
        # Generate Qdrant point IDs
        qdrant_ids = []
        points = []
        
        for point_id, vector, meta in zip(chunk_ids, embeddings, metadata):
            # Generate a unique Qdrant ID that includes both chunk ID and filename
            qdrant_id = hashlib.md5(
                f"{meta['filename']}-{point_id}-{meta['chunk_number']}"
                .encode()
            ).hexdigest()
            
            qdrant_ids.append(qdrant_id)
            points.append({
                "id": qdrant_id,
                "vector": vector,
                "payload": {
                    **meta,
                    "chunk_id": point_id,  # Store original chunk ID in payload
                    "updated_at": datetime.now().isoformat()
                }
            })
        
        # Upload to Qdrant
        qdrant_url = os.environ["QDRANT_URL"]
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        response = requests.put(
            f"{qdrant_url}/collections/{collection_name}/points",
            headers=headers,
            json={"points": points}
        )
        response.raise_for_status()
        
        # Log success
        log_qdrant_operation("Upload Success", {
            'request_id': request_id,
            'points_uploaded': len(points),
            'response': response.json() if response.text else {}
        })
        
        return qdrant_ids
        
    except Exception as e:
        # Log failure
        log_qdrant_operation("Upload Failed", {
            'request_id': request_id,
            'error': str(e)
        }, success=False)
        raise

def upload_vectors(vectors: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
    """
    Upload vectors to Qdrant using REST API.
    
    Args:
        vectors: List of vectors to upload
        metadata_list: List of metadata dictionaries
        
    Returns:
        List of point IDs
    """
    url = f"{os.getenv('QDRANT_URL')}/collections/{os.getenv('QDRANT_COLLECTION_NAME')}/points"
    headers = {
        "api-key": os.getenv('QDRANT_API_KEY'),
        "Content-Type": "application/json"
    }
    
    # Prepare points
    points = []
    for vector, metadata in zip(vectors, metadata_list):
        point = {
            "vector": vector,
            "payload": metadata
        }
        points.append(point)
        
    # Batch upload points
    try:
        response = requests.put(url, headers=headers, json={"points": points})
        response.raise_for_status()
        
        result = response.json()
        if not result.get('result', {}).get('status') == 'completed':
            raise ValueError(f"Upload failed: {result}")
            
        # Return point IDs
        return [str(i) for i in range(len(points))]
        
    except Exception as e:
        logging.error(f"Error uploading vectors to Qdrant: {str(e)}")
        raise

def verify_vectors(point_ids: List[str]) -> List[bool]:
    """
    Verify vectors exist in Qdrant.
    
    Args:
        point_ids: List of point IDs to verify
        
    Returns:
        List of booleans indicating if each point exists
    """
    url = f"{os.getenv('QDRANT_URL')}/collections/{os.getenv('QDRANT_COLLECTION_NAME')}/points"
    headers = {
        "api-key": os.getenv('QDRANT_API_KEY'),
        "Content-Type": "application/json"
    }
    
    try:
        # Get points in batches
        batch_size = 100
        results = []
        
        for i in range(0, len(point_ids), batch_size):
            batch = point_ids[i:i + batch_size]
            
            # Prepare request
            params = {
                "ids": batch,
                "with_payload": False,
                "with_vector": False
            }
            
            # Make request
            response = requests.post(f"{url}/get", headers=headers, json=params)
            response.raise_for_status()
            
            # Process response
            result = response.json()
            found_ids = {str(point['id']) for point in result.get('result', [])}
            
            # Mark each ID as found or not
            results.extend(id in found_ids for id in batch)
            
        return results
        
    except Exception as e:
        logging.error(f"Error verifying vectors in Qdrant: {str(e)}")
        raise

def search_vectors(query_vector: List[float], limit: int = 10, filter_params: Optional[Dict] = None) -> List[Dict]:
    """
    Search for similar vectors in Qdrant.
    
    Args:
        query_vector: Vector to search for
        limit: Maximum number of results
        filter_params: Optional filter parameters
        
    Returns:
        List of search results with scores and metadata
    """
    url = f"{os.getenv('QDRANT_URL')}/collections/{os.getenv('QDRANT_COLLECTION_NAME')}/points/search"
    headers = {
        "api-key": os.getenv('QDRANT_API_KEY'),
        "Content-Type": "application/json"
    }
    
    # Prepare search params
    params = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True
    }
    if filter_params:
        params["filter"] = filter_params
        
    try:
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        
        result = response.json()
        return result.get('result', [])
        
    except Exception as e:
        logging.error(f"Error searching vectors in Qdrant: {str(e)}")
        raise

@with_retries
def upload_to_qdrant(chunk_id: str, vector: List[float], metadata: Dict[str, Any]) -> str:
    """
    Upload a vector to Qdrant with retry logic using HTTP API.
    
    Args:
        chunk_id: ID of the chunk
        vector: Embedding vector
        metadata: Metadata for the vector
        
    Returns:
        Qdrant point ID
        
    Raises:
        QdrantError: If upload fails
    """
    try:
        # Validate inputs
        validate_vector_data([vector], "Upload")
        validate_metadata(metadata, "Upload")
        
        # Log operation start
        request_id = log_qdrant_operation("Upload", {
            'chunk_id': chunk_id,
            'metadata': metadata
        })
        
        # Prepare request
        qdrant_url = os.environ["QDRANT_URL"]
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        headers = {
            "api-key": os.environ["QDRANT_API_KEY"],
            "Content-Type": "application/json"
        }
        
        # Prepare payload
        payload = {
            "points": [{
                "id": chunk_id,
                "vector": vector,
                "payload": {
                    **metadata,
                    "chunk_id": chunk_id,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }]
        }
        
        # Make request
        response = requests.put(
            f"{qdrant_url}/collections/{collection_name}/points",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Log success
        log_qdrant_operation("Upload Success", {
            'request_id': request_id,
            'chunk_id': chunk_id,
            'response': response.json() if response.text else {}
        })
        
        return chunk_id
        
    except Exception as e:
        # Log failure
        log_qdrant_operation("Upload Failed", {
            'chunk_id': chunk_id,
            'error': str(e)
        }, success=False)
        raise QdrantError(f"Failed to upload vector: {str(e)}")