"""
Document processing pipeline using SQLAlchemy models.
"""

import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import Document, Chunk, ProcessedFile, ProcessingHistory
from src.database.session import get_db
from src.api.jina import generate_embedding
from src.api.qdrant import upload_to_qdrant
from src.processing.documents import chunk_document

logger = logging.getLogger(__name__)

def process_document(session: Session, doc_path: str, chunk_texts: List[str], token_counts: List[int]) -> bool:
    """
    Process a document and its chunks.
    
    Args:
        session: SQLAlchemy session
        doc_path: Path to the document
        chunk_texts: List of chunk texts
        token_counts: List of token counts for each chunk
        
    Returns:
        bool: True if successful
    """
    try:
        # Get filename
        filename = Path(doc_path).name
        
        # Create document record
        doc_id = hashlib.md5(doc_path.encode()).hexdigest()
        doc = Document(
            id=doc_id,
            filename=filename,
            status="processing",
            processed_at=datetime.utcnow()
        )
        session.add(doc)
        
        # Create chunks
        for i, (text, token_count) in enumerate(zip(chunk_texts, token_counts)):
            chunk_id = hashlib.md5(f"{doc_id}-{i}".encode()).hexdigest()
            content_hash = hashlib.md5(text.encode()).hexdigest()
            chunk = Chunk(
                id=chunk_id,
                document_id=doc_id,
                filename=filename,
                content=text,
                chunk_number=i,
                token_count=token_count,
                content_hash=content_hash,
                chunking_status="completed",
                embedding_status="pending",
                qdrant_status="pending",
                created_at=datetime.utcnow()
            )
            session.add(chunk)
        
        # Create processed file record
        processed_file = ProcessedFile(
            filename=filename,
            chunk_count=len(chunk_texts),
            status="processing",
            processed_at=datetime.utcnow()
        )
        session.add(processed_file)
        
        session.commit()
        logger.info(f"Successfully processed document {filename} with {len(chunk_texts)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document {doc_path}: {e}", exc_info=True)
        session.rollback()
        return False

def _update_document_status(session: Session, doc_id: str) -> None:
    """
    Update document status based on its chunks' statuses.
    
    Args:
        session: SQLAlchemy session
        doc_id: Document ID to update
    """
    # Get all chunks for the document
    chunks = session.query(Chunk).filter_by(document_id=doc_id).all()
    if not chunks:
        return
        
    # Check chunk statuses
    all_completed = all(
        chunk.chunking_status == "completed" and
        chunk.embedding_status == "completed" and
        chunk.qdrant_status == "completed"
        for chunk in chunks
    )
    
    any_failed = any(
        chunk.chunking_status == "failed" or
        chunk.embedding_status == "failed" or
        chunk.qdrant_status == "failed"
        for chunk in chunks
    )
    
    # Update document status
    doc = session.query(Document).filter_by(id=doc_id).first()
    if doc:
        if all_completed:
            doc.status = "completed"
            doc.chunking_status = "completed"
            doc.embedding_status = "completed"
            doc.qdrant_status = "completed"
        elif any_failed:
            doc.status = "failed"
            # Keep individual statuses for debugging
        else:
            doc.status = "processing"
        
        doc.processed_at = datetime.utcnow()
        
        # Update processed_files record
        processed_file = session.query(ProcessedFile).filter_by(filename=doc.filename).first()
        if processed_file:
            processed_file.status = doc.status
            processed_file.processed_at = doc.processed_at
            processed_file.chunking_status = doc.chunking_status
            processed_file.embedding_status = doc.embedding_status
            processed_file.qdrant_status = doc.qdrant_status
        
        # Log to processing history
        history = ProcessingHistory(
            document_id=doc_id,
            action="status_update",
            status=doc.status,
            details={
                "chunking_status": doc.chunking_status,
                "embedding_status": doc.embedding_status,
                "qdrant_status": doc.qdrant_status
            }
        )
        session.add(history)
        
        session.commit()

def process_pending_chunks(session: Session = None) -> Dict[str, int]:
    """
    Process pending chunks by generating embeddings and uploading to Qdrant.
    
    Args:
        session: Optional SQLAlchemy session
        
    Returns:
        Dict with processed and failed counts
    """
    if session is None:
        with get_db() as session:
            return _process_pending_chunks(session)
    else:
        return _process_pending_chunks(session)

def _process_pending_chunks(session: Session) -> Dict[str, int]:
    """Internal function to process pending chunks."""
    stats = {"processed": 0, "failed": 0}
    
    try:
        # Get chunks pending embedding
        pending_chunks = session.query(Chunk).filter(
            Chunk.embedding_status == "pending",
            Chunk.chunking_status == "completed"
        ).all()
        
        # Group chunks by document for efficient status updates
        doc_chunks = {}
        for chunk in pending_chunks:
            if chunk.document_id not in doc_chunks:
                doc_chunks[chunk.document_id] = []
            doc_chunks[chunk.document_id].append(chunk)
        
        for doc_id, chunks in doc_chunks.items():
            for chunk in chunks:
                try:
                    # Generate embedding
                    chunk.embedding_status = "processing"
                    session.commit()
                    
                    embedding = generate_embedding(chunk.content)
                    if embedding is None:
                        raise ValueError("Failed to generate embedding")
                    
                    # Convert embedding list to bytes for storage
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    chunk.embedding = embedding_bytes
                    chunk.embedding_status = "completed"
                    chunk.processed_at = datetime.utcnow()
                    session.commit()
                    
                    # Upload to Qdrant
                    chunk.qdrant_status = "processing"
                    session.commit()
                    
                    metadata = {
                        "filename": chunk.filename,
                        "chunk_number": chunk.chunk_number,
                        "document_id": chunk.document_id
                    }
                    
                    qdrant_id = upload_to_qdrant(chunk.id, embedding, metadata)
                    if not qdrant_id:
                        raise ValueError("Failed to upload to Qdrant")
                    
                    chunk.qdrant_id = qdrant_id
                    chunk.qdrant_status = "completed"
                    chunk.processed_at = datetime.utcnow()
                    session.commit()
                    
                    stats["processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.id}: {e}")
                    chunk.embedding_status = "failed"
                    chunk.qdrant_status = "failed"
                    chunk.error_message = str(e)
                    chunk.processed_at = datetime.utcnow()
                    session.commit()
                    stats["failed"] += 1
            
            # Update document status after processing all its chunks
            _update_document_status(session, doc_id)
            
        return stats
        
    except Exception as e:
        logger.error(f"Error in process_pending_chunks: {e}")
        session.rollback()
        raise

def display_processing_stats(stats: Dict[str, Any]) -> None:
    """Display processing statistics in a formatted way."""
    print("\nProcessing Statistics:")
    print("-" * 50)
    
    # Document statistics
    print("\nDocument Status:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Completed: {stats['completed_documents']}")
    print(f"Failed: {stats['failed_documents']}")
    print(f"Processing: {stats.get('processing_documents', 0)}")
    
    # Chunk statistics
    print("\nChunk Status:")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Completed: {stats['completed_chunks']}")
    print(f"Failed: {stats['failed_chunks']}")
    print(f"Pending: {stats.get('pending_chunks', 0)}")
    
    # Processing time
    if 'avg_processing_time' in stats:
        print(f"\nAverage Processing Time: {stats['avg_processing_time']:.2f} seconds")
    
    print("-" * 50)

def get_processing_stats(session: Session) -> Dict[str, Any]:
    """Get current processing statistics."""
    try:
        stats = {
            "total_documents": session.query(Document).count(),
            "completed_documents": session.query(Document).filter_by(status="completed").count(),
            "failed_documents": session.query(Document).filter_by(status="failed").count(),
            "processing_documents": session.query(Document).filter_by(status="processing").count(),
            "total_chunks": session.query(Chunk).count(),
            "completed_chunks": session.query(Chunk).filter(
                Chunk.embedding_status == "completed",
                Chunk.qdrant_status == "completed"
            ).count(),
            "failed_chunks": session.query(Chunk).filter(
                (Chunk.embedding_status == "failed") | (Chunk.qdrant_status == "failed")
            ).count(),
            "pending_chunks": session.query(Chunk).filter(
                (Chunk.embedding_status == "pending") | (Chunk.qdrant_status == "pending")
            ).count()
        }
        
        # Add average processing times
        completed_chunks = session.query(Chunk).filter(
            Chunk.embedding_status == "completed",
            Chunk.qdrant_status == "completed"
        ).all()
        
        if completed_chunks:
            total_time = sum(
                (chunk.updated_at - chunk.created_at).total_seconds()
                for chunk in completed_chunks
            )
            stats["avg_processing_time"] = total_time / len(completed_chunks)
        else:
            stats["avg_processing_time"] = 0
            
        # Display the statistics
        display_processing_stats(stats)
            
        return stats
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        return {}

def process_documents(session: Session, document_ids: List[str]) -> Dict[str, int]:
    """
    Process a batch of documents.
    
    Args:
        session: Database session
        document_ids: List of document IDs to process
        
    Returns:
        Dict with processing results
    """
    try:
        processed = 0
        failed = 0
        
        for doc_id in document_ids:
            try:
                document = session.query(Document).filter_by(id=doc_id).first()
                if not document:
                    logger.error(f"Document {doc_id} not found")
                    failed += 1
                    continue
                
                # Process document chunks
                if process_document_chunks(session, document):
                    processed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                failed += 1
                
        # Get and display final statistics
        stats = get_processing_stats(session)
        logger.info(f"Processing completed. Processed: {processed}, Failed: {failed}")
                
        return {"processed": processed, "failed": failed}
        
    except Exception as e:
        logger.error(f"Error in process_documents: {e}")
        return {"processed": 0, "failed": 1}