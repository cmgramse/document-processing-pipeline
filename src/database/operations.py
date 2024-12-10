"""
Database operations using SQLAlchemy models.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, case
from .models import Document, Chunk, ProcessedFile, ProcessingHistory
from .session import get_db
from src.api.qdrant import cleanup_orphaned_vectors, upload_to_qdrant
from src.api.jina import generate_embedding
import numpy as np

logger = logging.getLogger(__name__)

def mark_file_as_processed(session: Session, filename: str, chunk_count: int) -> None:
    """
    Mark a file as processed in the database.
    
    Args:
        session: SQLAlchemy session
        filename: Path of the file to mark as processed
        chunk_count: Number of chunks created from the file
    """
    try:
        basename = Path(filename).name
        
        # Update document status
        document = session.query(Document).filter_by(filename=basename).first()
        if document:
            document.status = 'completed'
            document.processed_at = datetime.utcnow()
        
        # Update or create processed file record
        processed_file = session.query(ProcessedFile).filter_by(filename=basename).first()
        if not processed_file:
            processed_file = ProcessedFile(
                filename=basename,
                chunk_count=chunk_count,
                status='completed',
                processed_at=datetime.utcnow()
            )
            session.add(processed_file)
        else:
            processed_file.status = 'completed'
            processed_file.chunk_count = chunk_count
            processed_file.processed_at = datetime.utcnow()
        
        session.commit()
        logger.info(f"Marked {filename} as processed with {chunk_count} chunks")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error marking file as processed: {str(e)}")
        raise

def force_reprocess_files(session: Session, filenames: List[str]) -> None:
    """
    Force reprocessing of specific files by removing their records.
    
    Args:
        session: SQLAlchemy session
        filenames: List of file paths to reprocess
    """
    try:
        for filename in filenames:
            basename = Path(filename).name
            
            # Delete from all tables
            session.query(Document).filter_by(filename=basename).delete()
            session.query(Chunk).filter_by(filename=basename).delete()
            session.query(ProcessedFile).filter_by(filename=basename).delete()
            
            logger.info(f"Cleared records for {filename}")
        
        session.commit()
        logger.info(f"Successfully marked {len(filenames)} files for reprocessing")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error marking files for reprocessing: {str(e)}")
        raise

def track_document_chunk(
    session: Session,
    filename: str,
    chunk_text: str,
    chunk_number: int,
    token_count: Optional[int] = None
) -> str:
    """
    Track a new document chunk in the database.
    
    Args:
        session: SQLAlchemy session
        filename: Source document filename
        chunk_text: Content of the chunk
        chunk_number: Sequence number of the chunk
        token_count: Optional count of tokens in chunk
        
    Returns:
        Chunk ID
    """
    try:
        basename = Path(filename).name
        
        # Generate chunk ID and content hash
        chunk_id = hashlib.md5(f"{basename}-{chunk_number}-{chunk_text[:100]}".encode()).hexdigest()
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
        
        # Create document if it doesn't exist
        doc_id = hashlib.md5(basename.encode()).hexdigest()
        document = session.query(Document).filter_by(id=doc_id).first()
        if not document:
            document = Document(
                id=doc_id,
                filename=basename,
                status='processing'
            )
            session.add(document)
        
        # Create chunk
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            filename=basename,
            content=chunk_text,
            token_count=token_count,
            chunk_number=chunk_number,
            content_hash=content_hash,
            chunking_status='completed'
        )
        session.add(chunk)
        
        session.commit()
        return chunk_id
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error tracking chunk: {str(e)}")
        raise

def get_chunk_embedding(session: Session, chunk_id: str) -> Optional[List[float]]:
    """
    Retrieve a chunk's embedding from the database.
    
    Args:
        session: SQLAlchemy session
        chunk_id: ID of the chunk
        
    Returns:
        List of floats if embedding exists, None otherwise
    """
    chunk = session.query(Chunk).filter_by(id=chunk_id).first()
    if chunk and chunk.embedding:
        # Convert bytes back to float list
        embedding_array = np.frombuffer(chunk.embedding, dtype=np.float32)
        return embedding_array.tolist()
    return None

def get_pending_chunks(session: Session, batch_size: int = 50) -> List[Chunk]:
    """Get chunks that need processing."""
    return session.query(Chunk).filter(
        and_(
            Chunk.chunking_status == 'completed',
            or_(
                Chunk.embedding_status == 'pending',
                Chunk.qdrant_status == 'pending'
            )
        )
    ).limit(batch_size).all()

def update_chunk_status(
    session: Session,
    chunk_id: str,
    status: str,
    **kwargs
) -> None:
    """Update chunk status with validation."""
    chunk = session.query(Chunk).filter_by(id=chunk_id).one()
    
    # Update status fields
    if 'embedding_status' in kwargs:
        chunk.embedding_status = kwargs['embedding_status']
    if 'qdrant_status' in kwargs:
        chunk.qdrant_status = kwargs['qdrant_status']
    
    # Update data fields
    if 'embedding' in kwargs:
        chunk.embedding = kwargs['embedding']
    if 'qdrant_id' in kwargs:
        chunk.qdrant_id = kwargs['qdrant_id']
    
    # Update metadata
    chunk.processed_at = datetime.utcnow()
    if 'error_message' in kwargs:
        chunk.error_message = kwargs['error_message']
    if 'last_verified_at' in kwargs:
        chunk.last_verified_at = kwargs['last_verified_at']
    
    # Log the update
    log_processing_history(
        session=session,
        chunk_id=chunk_id,
        action='update_status',
        status=status,
        details=kwargs
    )

def get_document_status(session: Session, file_path: str) -> Optional[Dict[str, Any]]:
    """Check if a document exists and get its status."""
    basename = Path(file_path).name
    doc = session.query(Document).filter_by(filename=basename).first()
    if not doc:
        return None
    
    return {
        'id': doc.id,
        'status': doc.status,
        'embedding_status': doc.embedding_status,
        'qdrant_status': doc.qdrant_status,
        'error_message': doc.error_message,
        'processed_at': doc.processed_at
    }

def get_unprocessed_files(session: Session, filenames: List[str]) -> List[str]:
    """
    Get list of files that haven't been processed yet.
    
    Args:
        session: SQLAlchemy session
        filenames: List of filenames to check
        
    Returns:
        List of unprocessed filenames
    """
    try:
        # Convert filenames to basenames
        basenames = [Path(f).name for f in filenames]
        
        # Get processed files
        processed = {
            row.filename for row in 
            session.query(ProcessedFile.filename)
            .filter(ProcessedFile.filename.in_(basenames))
            .filter(ProcessedFile.status == 'completed')
            .all()
        }
        
        # Return files not in processed set
        unprocessed = []
        for filename, basename in zip(filenames, basenames):
            if basename not in processed:
                unprocessed.append(filename)
                
        logger.info(f"Found {len(unprocessed)} unprocessed files out of {len(filenames)}")
        return unprocessed
        
    except Exception as e:
        logger.error(f"Error getting unprocessed files: {str(e)}")
        raise

def delete_document(session: Session, filename: str) -> bool:
    """Delete a document and its chunks."""
    try:
        # Get document and its chunks
        doc = session.query(Document).filter_by(filename=filename).first()
        if not doc:
            return False
        
        # Get Qdrant IDs before deletion for cleanup
        chunks = session.query(Chunk).filter_by(document_id=doc.id).all()
        qdrant_ids = [c.qdrant_id for c in chunks if c.qdrant_id]
        
        # Delete from database
        session.delete(doc)  # This will cascade to chunks
        session.query(ProcessedFile).filter_by(filename=filename).delete()
        
        # Cleanup Qdrant vectors
        if qdrant_ids:
            cleanup_orphaned_vectors(qdrant_ids)
        
        session.commit()
        logger.info(f"Successfully deleted document {filename}")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to delete document {filename}: {e}")
        return False

def get_system_stats(session: Session) -> Dict[str, Any]:
    """Get comprehensive system statistics."""
    stats = {}
    
    # Document stats
    doc_stats = session.query(
        func.count(Document.id).label('total'),
        func.sum(case([(Document.status == 'completed', 1)], else_=0)).label('completed'),
        func.sum(case([(Document.status == 'failed', 1)], else_=0)).label('failed')
    ).first()
    
    stats.update({
        'total_documents': doc_stats.total or 0,
        'completed_documents': doc_stats.completed or 0,
        'failed_documents': doc_stats.failed or 0
    })
    
    # Chunk stats
    chunk_stats = session.query(
        func.count(Chunk.id).label('total'),
        func.sum(case([(and_(Chunk.embedding_status == 'completed',
                            Chunk.qdrant_status == 'completed'), 1)], else_=0)).label('completed'),
        func.sum(case([(or_(Chunk.embedding_status == 'failed',
                           Chunk.qdrant_status == 'failed'), 1)], else_=0)).label('failed')
    ).first()
    
    stats.update({
        'total_chunks': chunk_stats.total or 0,
        'completed_chunks': chunk_stats.completed or 0,
        'failed_chunks': chunk_stats.failed or 0
    })
    
    # Recent activity
    recent_history = session.query(ProcessingHistory)\
        .order_by(ProcessingHistory.created_at.desc())\
        .limit(10)\
        .all()
    
    stats['recent_activity'] = [
        {
            'action': h.action,
            'status': h.status,
            'details': h.details,
            'created_at': h.created_at
        }
        for h in recent_history
    ]
    
    return stats

def cleanup_database(session: Session, retention_days: int = 30) -> Dict[str, int]:
    """Clean up old and failed entries."""
    stats = {'cleaned_documents': 0, 'cleaned_chunks': 0}
    
    try:
        # Clean up old documents
        old_date = datetime.utcnow() - timedelta(days=retention_days)
        old_docs = session.query(Document)\
            .filter(Document.processed_at < old_date)\
            .all()
        
        for doc in old_docs:
            delete_document(session, doc.filename)
            stats['cleaned_documents'] += 1
        
        # Clean up failed chunks
        failed_chunks = session.query(Chunk)\
            .filter(or_(
                Chunk.embedding_status == 'failed',
                Chunk.qdrant_status == 'failed'
            ))\
            .all()
        
        for chunk in failed_chunks:
            if chunk.error_message and 'permanent failure' in chunk.error_message.lower():
                session.delete(chunk)
                stats['cleaned_chunks'] += 1
        
        session.commit()
        return stats
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to clean up database: {e}")
        raise

def sync_with_qdrant(session: Session) -> bool:
    """Synchronize database status with Qdrant."""
    try:
        # Get all chunks with Qdrant IDs
        chunks = session.query(Chunk)\
            .filter(Chunk.qdrant_id.isnot(None))\
            .all()
        
        # Verify each chunk in Qdrant
        for chunk in chunks:
            if not verify_qdrant_vector(chunk.qdrant_id):
                chunk.qdrant_status = 'failed'
                chunk.error_message = 'Vector not found in Qdrant'
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to sync with Qdrant: {e}")
        return False

def log_processing_history(
    session: Session,
    chunk_id: Optional[str] = None,
    document_id: Optional[str] = None,
    action: str = '',
    status: str = '',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log processing action to history."""
    history = ProcessingHistory(
        chunk_id=chunk_id,
        document_id=document_id,
        action=action,
        status=status,
        details=details or {}
    )
    session.add(history)

def verify_chunk_state(chunk: Chunk) -> bool:
    """Verify chunk state consistency."""
    try:
        # This will trigger the validation
        if chunk.qdrant_status == 'completed':
            chunk.validate_qdrant_status('qdrant_status', 'completed')
        return True
    except ValueError as e:
        logger.warning(f"Chunk {chunk.id} state verification failed: {str(e)}")
        return False

def get_processing_stats(session: Session) -> Dict[str, Any]:
    """Get current processing statistics."""
    stats = {}
    
    # Chunk processing stats
    chunk_stats = session.query(
        func.count(Chunk.id).label('total_chunks'),
        func.sum(case([(Chunk.embedding_status == 'pending', 1)], else_=0)).label('pending_chunks'),
        func.sum(case([(Chunk.embedding_status == 'processing', 1)], else_=0)).label('processing_chunks'),
        func.sum(case([(Chunk.embedding_status == 'completed', 1)], else_=0)).label('completed_chunks'),
        func.sum(case([(Chunk.embedding_status == 'failed', 1)], else_=0)).label('failed_chunks')
    ).first()
    
    stats.update({
        'total_chunks': chunk_stats.total_chunks or 0,
        'pending_chunks': chunk_stats.pending_chunks or 0,
        'processing_chunks': chunk_stats.processing_chunks or 0,
        'completed_chunks': chunk_stats.completed_chunks or 0,
        'failed_chunks': chunk_stats.failed_chunks or 0
    })
    
    # Document processing stats
    doc_stats = session.query(
        func.count(Document.id).label('total_docs'),
        func.sum(case([(Document.status == 'pending', 1)], else_=0)).label('pending_docs'),
        func.sum(case([(Document.status == 'processing', 1)], else_=0)).label('processing_docs'),
        func.sum(case([(Document.status == 'completed', 1)], else_=0)).label('completed_docs'),
        func.sum(case([(Document.status == 'failed', 1)], else_=0)).label('failed_docs')
    ).first()
    
    stats.update({
        'total_documents': doc_stats.total_docs or 0,
        'pending_documents': doc_stats.pending_docs or 0,
        'processing_documents': doc_stats.processing_docs or 0,
        'completed_documents': doc_stats.completed_docs or 0,
        'failed_documents': doc_stats.failed_docs or 0
    })
    
    # Recent activity
    recent_activity = session.query(ProcessingHistory)\
        .order_by(ProcessingHistory.created_at.desc())\
        .limit(5)\
        .all()
    
    stats['recent_activity'] = [
        {
            'created_at': activity.created_at,
            'action': activity.action,
            'status': activity.status,
            'details': activity.details
        }
        for activity in recent_activity
    ]
    
    return stats
def cleanup_failed_chunks(session: Session) -> int:
    """Reset failed chunks to pending status."""
    failed_chunks = session.query(Chunk).filter(
        or_(
            Chunk.embedding_status == 'failed',
            Chunk.qdrant_status == 'failed'
        )
    ).all()
    
    count = 0
    for chunk in failed_chunks:
        if chunk.embedding_status == 'failed':
            chunk.embedding_status = 'pending'
        if chunk.qdrant_status == 'failed':
            chunk.qdrant_status = 'pending'
        chunk.error_message = None
        count += 1
    
    return count

def process_pending_chunks(session: Session, batch_size: int = 50) -> Tuple[int, int]:
    """
    Process all pending chunks in the database.
    
    Args:
        session: SQLAlchemy session
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    processed_count = 0
    error_count = 0
    
    try:
        while True:
            # Get next batch of pending chunks
            chunks = session.query(Chunk).filter(
                or_(
                    Chunk.embedding_status == 'pending',
                    and_(
                        Chunk.embedding_status == 'failed',
                        Chunk.version < 3  # Retry up to 3 times
                    )
                )
            ).limit(batch_size).all()
            
            if not chunks:
                break
                
            try:
                # Process chunks
                for chunk in chunks:
                    try:
                        # Generate embedding
                        embedding = generate_embedding(chunk.content)
                        
                        # Upload to Qdrant
                        qdrant_id = upload_to_qdrant(
                            chunk.id,
                            embedding,
                            {
                                'filename': chunk.filename,
                                'chunk_number': chunk.chunk_number
                            }
                        )
                        
                        # Update chunk
                        chunk.embedding = embedding
                        chunk.embedding_status = 'completed'
                        chunk.qdrant_id = qdrant_id
                        chunk.qdrant_status = 'completed'
                        chunk.processed_at = datetime.utcnow()
                        chunk.version += 1
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process chunk {chunk.id}: {e}")
                        chunk.embedding_status = 'failed'
                        chunk.qdrant_status = 'failed'
                        chunk.error_message = str(e)
                        chunk.version += 1
                        error_count += 1
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Batch processing failed: {e}")
                error_count += len(chunks)
                
                # Mark all chunks as failed
                for chunk in chunks:
                    chunk.embedding_status = 'failed'
                    chunk.qdrant_status = 'failed'
                    chunk.error_message = str(e)
                    chunk.version += 1
                
                session.commit()
                
    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        raise
        
    return processed_count, error_count
