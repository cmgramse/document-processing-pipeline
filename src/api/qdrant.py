"""
Qdrant vector database integration module.

This module provides functionality for interacting with the Qdrant vector database,
including initialization, vector uploads, searches, and deletions. It handles
connection management, error recovery, and operation logging.

The module requires the following environment variables:
    - QDRANT_API_KEY: API key for Qdrant authentication
    - QDRANT_URL: URL of the Qdrant server
    - QDRANT_COLLECTION_NAME: Name of the collection to use

Example:
    Initialize Qdrant connection:
        client = QdrantClient()
    
    Upload vectors:
        success = client.upload_vectors(vectors, metadata)
    
    Search similar documents:
        results = client.similarity_search(query, k=5)
"""

import logging
import hashlib
from datetime import datetime
import requests
import os
import json
import time
from typing import Dict, Any, List
from langchain.schema import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import Qdrant

def log_qdrant_operation(operation: str, details: Dict[str, Any], success: bool = True) -> str:
    """
    Log Qdrant operations for monitoring and debugging.
    
    Args:
        operation: Name of the operation being performed
        details: Dictionary containing operation details
        success: Whether the operation was successful
    
    The function logs:
        - Timestamp
        - Operation name
        - Success status
        - Operation details
        - Any error information
    
    Returns:
        str: Request ID for the logged operation
    """
    api_logger = logging.getLogger('api_calls')
    status = "SUCCESS" if success else "FAILED"
    request_id = hashlib.md5(f"{datetime.now()}-qdrant-{operation}".encode()).hexdigest()[:8]
    
    api_logger.info(
        f"[{request_id}] Qdrant {operation} {status} - "
        f"Collection: {os.environ['QDRANT_COLLECTION_NAME']} | "
        f"Details: {json.dumps(details)}"
    )
    return request_id

def validate_qdrant_connection() -> bool:
    """
    Validate connection to Qdrant server.
    
    Attempts to establish a connection to the Qdrant server using environment
    variables. Performs basic operations to ensure the connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    
    Raises:
        Exception: If connection fails
    """
    try:
        qdrant_url = os.environ["QDRANT_URL"]
        request_id = log_qdrant_operation("Connection Test", {
            "url": qdrant_url,
            "collection": os.environ['QDRANT_COLLECTION_NAME']
        })
        
        response = requests.get(
            f"{qdrant_url}/collections",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        response.raise_for_status()
        
        log_qdrant_operation("Connection Success", {
            "status_code": response.status_code,
            "collections_available": True
        })
        return True
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "url": qdrant_url
        }
        log_qdrant_operation("Connection Failed", error_details, success=False)
        logging.error(f"Failed to connect to Qdrant: {str(e)}")
        return False

class QdrantClient:
    """Client for interacting with Qdrant vector database."""
    
    def __init__(self):
        """Initialize Qdrant client with environment variables."""
        self.url = os.environ["QDRANT_URL"]
        self.api_key = os.environ["QDRANT_API_KEY"]
        self.collection_name = os.environ["QDRANT_COLLECTION_NAME"]

    def upload_vectors(self, vectors: List[Any], metadata: Dict[str, Any]) -> bool:
        """Uploads a list of vectors and their corresponding metadata to Qdrant."""
        request_id = log_qdrant_operation("Vector Upload", {
            "num_vectors": len(vectors),
            "metadata": metadata
        })
        
        try:
            response = requests.post(
                f"{self.url}/collections/{self.collection_name}/vectors",
                headers={"api-key": self.api_key},
                json=vectors
            )
            response.raise_for_status()
            
            log_qdrant_operation("Vector Upload Success", {
                "status_code": response.status_code,
                "num_vectors_uploaded": len(vectors)
            })
            return True
        
        except Exception as e:
            error_details = {
                "error": str(e),
                "num_vectors": len(vectors)
            }
            log_qdrant_operation("Vector Upload Failed", error_details, success=False)
            logging.error(f"Failed to upload vectors: {str(e)}")
            return False

    def similarity_search(self, query: List[Any], k: int = 5) -> Dict[str, Any]:
        """Performs a similarity search on Qdrant with the given query vector."""
        request_id = log_qdrant_operation("Similarity Search", {
            "query_vector": query,
            "k": k
        })
        
        try:
            response = requests.post(
                f"{self.url}/collections/{self.collection_name}/search",
                headers={"api-key": self.api_key},
                json={"vectors": [query], "k": k}
            )
            response.raise_for_status()
            
            log_qdrant_operation("Similarity Search Success", {
                "status_code": response.status_code,
                "num_results_found": len(response.json()["results"])
            })
            return response.json()["results"]
        
        except Exception as e:
            error_details = {
                "error": str(e),
                "query": query,
                "k": k
            }
            log_qdrant_operation("Search Failed", error_details, success=False)
            logging.error(f"Failed to perform similarity search: {str(e)}")
            raise

def initialize_qdrant(conn, max_retries: int = 3) -> Qdrant:
    """
    Initialize Qdrant vector store with pending documents.
    
    Retrieves pending documents from the SQLite database and uploads them to
    Qdrant. Updates the document status after successful upload.
    
    Args:
        conn: SQLite database connection
        max_retries: Maximum number of retries for Qdrant initialization
    
    Returns:
        Qdrant: Initialized Qdrant vector store if successful
    
    Raises:
        Exception: If Qdrant initialization fails
    """
    api_logger = logging.getLogger('api_calls')
    
    try:
        # Get pending documents from SQLite
        c = conn.cursor()
        c.execute('''SELECT id, content, embedding, filename, chunk_id 
                    FROM documents 
                    WHERE qdrant_status = 'pending' ''')
        pending_docs = c.fetchall()
        
        if not pending_docs:
            api_logger.info("No pending documents to upload to Qdrant")
            return None
            
        # Convert to Document objects
        documents = []
        embeddings_list = []
        doc_ids = []
        
        for doc_id, content, embedding_json, filename, chunk_id in pending_docs:
            embedding = json.loads(embedding_json)
            doc = Document(
                page_content=content,
                metadata={
                    'source': filename,
                    'chunk_id': chunk_id
                }
            )
            documents.append(doc)
            embeddings_list.append(embedding)
            doc_ids.append(doc_id)
        
        # Initialize Qdrant client
        embeddings = JinaEmbeddings(
            jina_api_key=os.environ["JINA_API_KEY"],
            model_name="jina-embeddings-v2-base-en"
        )
        
        qdrant = Qdrant.from_embeddings(
            embeddings_list,
            embeddings,
            url=os.environ["QDRANT_URL"],
            prefer_grpc=True,
            api_key=os.environ["QDRANT_API_KEY"],
            collection_name=os.environ["QDRANT_COLLECTION_NAME"],
            force_recreate=False  # Never recreate to avoid losing existing data
        )
        
        # Update upload status in SQLite
        upload_time = datetime.now().isoformat()
        c.executemany('''UPDATE documents 
                        SET qdrant_status = 'uploaded', 
                            uploaded_at = ? 
                        WHERE id = ?''',
                     [(upload_time, doc_id) for doc_id in doc_ids])
        conn.commit()
        
        api_logger.info(f"Successfully uploaded {len(documents)} documents to Qdrant")
        return qdrant
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "documents_affected": len(pending_docs) if pending_docs else 0
        }
        log_qdrant_operation("Upload Failed", error_details, success=False)
        
        # Mark failed documents
        if pending_docs:
            c.executemany('''UPDATE documents 
                            SET qdrant_status = 'failed' 
                            WHERE id = ?''',
                         [(doc_id,) for doc_id, *_ in pending_docs])
            conn.commit()
        
        raise

def delete_vectors_by_filter(client: Qdrant, filter_dict: Dict[str, Any]) -> bool:
    """
    Delete vectors from Qdrant based on metadata filter.
    
    Args:
        client: Qdrant client instance
        filter_dict: Dictionary containing filter criteria
    
    Returns:
        bool: True if deletion successful, False otherwise
    
    Example:
        # Delete vectors for a specific document
        success = delete_vectors_by_filter(client, {
            'metadata': {'source': 'doc1.md'}
        })
    """
    try:
        request_id = log_qdrant_operation("Delete Vectors", {
            "filter": filter_dict,
            "collection": os.environ['QDRANT_COLLECTION_NAME']
        })
        
        client.delete(
            collection_name=os.environ['QDRANT_COLLECTION_NAME'],
            points_selector=filter_dict
        )
        
        log_qdrant_operation("Delete Success", {
            "request_id": request_id,
            "filter": filter_dict
        })
        return True
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "filter": filter_dict
        }
        log_qdrant_operation("Delete Failed", error_details, success=False)
        return False