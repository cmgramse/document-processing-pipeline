"""
SQLAlchemy models for the document processing pipeline.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, DateTime, Text, BLOB, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates

Base = declarative_base()

class Document(Base):
    """Document model representing original documents to be processed."""
    __tablename__ = 'documents'

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    content = Column(Text)
    embedding = Column(Text)
    processed_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='pending')
    chunking_status = Column(String, default='pending')
    embedding_status = Column(String, default='pending')
    qdrant_status = Column(String, default='pending')
    error_message = Column(Text)
    version = Column(Integer, default=1)

    # Relationships
    chunks = relationship("Chunk", back_populates="document")

    @validates('status', 'chunking_status', 'embedding_status', 'qdrant_status')
    def validate_status(self, key, status):
        valid_statuses = {'pending', 'processing', 'completed', 'failed', 'retrying'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status

class Chunk(Base):
    """Chunk model representing document segments with vector information."""
    __tablename__ = 'chunks'

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id', ondelete='CASCADE'))
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    chunk_number = Column(Integer, nullable=False)
    content_hash = Column(String, nullable=False)
    chunking_status = Column(String, default='pending')
    embedding_status = Column(String, default='pending')
    qdrant_status = Column(String, default='pending')
    embedding = Column(BLOB)
    qdrant_id = Column(String)
    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_verified_at = Column(DateTime)
    error_message = Column(Text)
    version = Column(Integer, default=1)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    @validates('chunking_status', 'embedding_status')
    def validate_status(self, key, status):
        valid_statuses = {'pending', 'processing', 'completed', 'failed', 'retrying'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status

    @validates('qdrant_status')
    def validate_qdrant_status(self, key, status):
        # First validate the status value
        valid_statuses = {'pending', 'processing', 'completed', 'failed', 'retrying'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        
        # Then check additional constraints for completed status
        if status == 'completed':
            if self.embedding_status != 'completed':
                raise ValueError("Cannot complete Qdrant status before embedding is completed")
            if not self.qdrant_id:
                raise ValueError("Cannot complete Qdrant status without qdrant_id")
        
        return status

class ProcessedFile(Base):
    """Tracks which files have been processed."""
    __tablename__ = 'processed_files'

    filename = Column(String, primary_key=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunk_count = Column(Integer)
    status = Column(String, default='pending')
    chunking_status = Column(String, default='pending')
    embedding_status = Column(String, default='pending')
    qdrant_status = Column(String, default='pending')
    last_verified_at = Column(DateTime)
    error_message = Column(Text)

    @validates('status', 'chunking_status', 'embedding_status', 'qdrant_status')
    def validate_status(self, key, status):
        valid_statuses = {'pending', 'processing', 'completed', 'failed', 'retrying'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of {valid_statuses}")
        return status

class ProcessingHistory(Base):
    """Logs all processing actions."""
    __tablename__ = 'processing_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey('documents.id', ondelete='CASCADE'))
    chunk_id = Column(String, ForeignKey('chunks.id', ondelete='CASCADE'))
    action = Column(String, nullable=False)
    status = Column(String, nullable=False)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document")
    chunk = relationship("Chunk") 