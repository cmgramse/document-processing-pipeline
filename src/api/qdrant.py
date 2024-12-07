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

def log_qdrant_operation(operation: str, details: Dict[str, Any], success: bool = True):
    """Log Qdrant operations with details"""
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
    """Validate Qdrant connection before processing"""
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

def initialize_qdrant(conn, max_retries: int = 3):
    """Initialize Qdrant with retry logic and verification"""
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