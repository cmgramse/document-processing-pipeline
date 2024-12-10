"""
Background processing tasks for document processing.
"""

import logging
import time
from typing import Optional
from datetime import datetime

from ..database.document_manager import DocumentManager
from ..database.connection import get_connection_pool, transaction, get_db_connection
from ..models.document import Document, ProcessingStats, ProcessingStatus
from ..api.jina import JinaAPI
from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

async def process_document_async(document_id: str) -> None:
    """
    Process a document asynchronously.
    
    This function is called by FastAPI's background tasks system.
    It handles the complete document processing pipeline:
    1. Load document from database
    2. Process content through Jina API
    3. Store vector embeddings in Qdrant
    4. Update document status and record processing stats
    """
    document_manager = DocumentManager()
    jina_api = JinaAPI()
    resource_monitor = ResourceMonitor()
    
    try:
        # Get document from database
        document = document_manager.get_document(document_id)
        if not document:
            logger.error(f"Document {document_id} not found")
            return
        
        # Initialize processing stats
        stats = ProcessingStats(document_id=document_id)
        stats.start()
        
        # Check resources before processing
        can_process, reason, resource_stats = resource_monitor.check_resources()
        if not can_process:
            logger.warning(f"Cannot process document {document_id}: {reason}")
            document.processing_status = ProcessingStatus.SKIPPED
            document.error_message = reason
            document_manager.update_document(document)
            stats.skipped_documents += 1
            return
        
        # Update resource stats
        stats.system_memory_percent = resource_stats['system_memory_percent']
        stats.process_memory_mb = resource_stats['process_memory_mb']
        
        try:
            # Update document status
            document.processing_status = ProcessingStatus.PROCESSING
            document_manager.update_document(document)
            
            # Process through Jina API
            embeddings = await jina_api.get_embeddings(document.content)
            vector_id = await jina_api.store_embeddings(embeddings, document_id)
            
            # Update document with vector ID and success status
            document.vector_id = vector_id
            document.processing_status = ProcessingStatus.COMPLETED
            document.error_message = None
            
            # Update stats
            stats.processed_documents += 1
            stats.total_documents += 1
            stats.total_embeddings += 1
            stats.tokens_processed = len(document.content.split())  # Simple token count
            stats.chunks_created = 1  # For now, we're not chunking
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            
            # Update stats
            stats.failed_documents += 1
            stats.total_documents += 1
            stats.failed_embeddings += 1
            stats.add_error(str(e))
            
            raise
        
        finally:
            # End stats tracking
            stats.end()
            
            # Update document and record stats in a single transaction
            pool = get_connection_pool()
            with get_db_connection(pool) as conn:
                with transaction(conn):
                    document_manager.update_document(document)
                    document_manager.record_processing_stats(stats)
            
            logger.info(
                f"Document {document_id} processed in {stats.processing_time:.2f}s "
                f"(status: {document.processing_status.value})"
            )
    
    except Exception as e:
        logger.error(f"Unhandled error processing document {document_id}: {str(e)}")
        # Ensure document is marked as failed
        try:
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = f"Unhandled error: {str(e)}"
            document_manager.update_document(document)
        except Exception as update_error:
            logger.error(
                f"Failed to update document {document_id} status: {str(update_error)}"
            )