"""
API routes for document processing.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.document import Document, ProcessingStats
from ..database.document_manager import DocumentManager
from ..processing.background_tasks import process_document_async

router = APIRouter()
document_manager = DocumentManager()

@router.post("/documents/", response_model=Document)
async def create_document(document: Document, background_tasks: BackgroundTasks):
    """Create a new document and queue it for processing."""
    try:
        document_manager.create_document(document)
        background_tasks.add_task(process_document_async, document.id)
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Retrieve a document by ID."""
    document = document_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.put("/documents/{document_id}", response_model=Document)
async def update_document(document_id: str, document: Document):
    """Update an existing document."""
    if document_id != document.id:
        raise HTTPException(status_code=400, detail="Document ID mismatch")
    
    existing = document_manager.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        document_manager.update_document(document)
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID."""
    if document_manager.delete_document(document_id):
        return {"status": "success", "message": "Document deleted"}
    raise HTTPException(status_code=404, detail="Document not found")

@router.get("/documents/", response_model=List[Document])
async def list_documents(status: Optional[str] = None):
    """List all documents, optionally filtered by processing status."""
    if status == "pending":
        return document_manager.get_pending_documents()
    return document_manager.get_all_documents()

@router.get("/documents/{document_id}/stats", response_model=List[ProcessingStats])
async def get_document_stats(document_id: str):
    """Get processing statistics for a document."""
    document = document_manager.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document_manager.get_processing_stats(document_id) 