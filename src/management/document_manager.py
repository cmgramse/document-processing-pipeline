"""Document management functionality for handling document lifecycle."""
import logging
import os
from typing import List, Tuple, Optional
from datetime import datetime
import hashlib
from pathlib import Path

from ..database.operations import get_database_stats
from ..api.qdrant import delete_vectors_by_filter

class DocumentManager:
    def __init__(self, conn, qdrant_client=None):
        self.conn = conn
        self.qdrant_client = qdrant_client
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME')
    
    def check_document_exists(self, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Check if document exists and its status.
        Returns (exists, status)
        """
        c = self.conn.cursor()
        c.execute("SELECT status FROM processed_files WHERE filename = ?", (filename,))
        result = c.fetchone()
        return (True, result[0]) if result else (False, None)
    
    def delete_document(self, filename: str, force: bool = False) -> bool:
        """Delete a document and its associated chunks."""
        try:
            c = self.conn.cursor()
            
            # Get all chunk IDs for the document
            c.execute("SELECT id FROM chunks WHERE filename = ?", (filename,))
            chunk_ids = [row[0] for row in c.fetchall()]
            
            # Get all document IDs in Qdrant
            c.execute("SELECT id FROM documents WHERE chunk_id IN ({})".format(
                ','.join('?' * len(chunk_ids))), chunk_ids)
            doc_ids = [row[0] for row in c.fetchall()]
            
            if not force:
                # Check if document is referenced elsewhere
                c.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='document_references'
                """)
                if c.fetchone():
                    c.execute("""
                        SELECT COUNT(*) FROM document_references 
                        WHERE source_doc_id IN (
                            SELECT id FROM documents WHERE filename = ?
                        ) OR target_doc_id IN (
                            SELECT id FROM documents WHERE filename = ?
                        )
                    """, (filename, filename))
                    ref_count = c.fetchone()[0]
                    if ref_count > 0:
                        logging.warning(f"Document {filename} has {ref_count} references")
                        if not force:
                            return False
            
            # Delete from Qdrant
            if self.qdrant_client and doc_ids:
                self.qdrant_client.delete_vectors_by_filter({"must": [{"id": {"in": doc_ids}}]})
            
            # Delete document references if table exists
            c.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='document_references'
            """)
            if c.fetchone():
                c.execute("""
                    DELETE FROM document_references 
                    WHERE source_doc_id IN (
                        SELECT id FROM documents WHERE filename = ?
                    ) OR target_doc_id IN (
                        SELECT id FROM documents WHERE filename = ?
                    )
                """, (filename, filename))
                
            # Delete from documents
            c.execute("DELETE FROM documents WHERE chunk_id IN ({})".format(
                ','.join('?' * len(chunk_ids))), chunk_ids)
            
            # Delete from chunks
            c.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
            
            # Delete from processed_files
            c.execute("DELETE FROM processed_files WHERE filename = ?", (filename,))
            
            self.conn.commit()
            logging.info(f"Successfully deleted document {filename} and all associated data")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting document {filename}: {str(e)}")
            self.conn.rollback()
            return False
    
    def select_documents(self, available_docs: List[str], selection: str) -> List[str]:
        """
        Select documents based on various input formats:
        - Individual numbers: "1,3,5"
        - Ranges: "1-5"
        - Combinations: "1,3-5,7"
        - All: "all"
        - Latest: "latest:N"
        """
        if not selection or not available_docs:
            return []
            
        selection = selection.lower().strip()
        selected = set()
        
        # Handle 'all' case
        if selection == 'all':
            return available_docs
            
        # Handle 'latest:N' case
        if selection.startswith('latest:'):
            try:
                n = int(selection.split(':')[1])
                return available_docs[-n:]
            except (IndexError, ValueError):
                logging.warning("Invalid 'latest:N' format, falling back to manual selection")
        
        # Handle individual numbers and ranges
        for part in selection.split(','):
            part = part.strip()
            try:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    if 1 <= start <= end <= len(available_docs):
                        selected.update(range(start-1, end))
                else:
                    idx = int(part) - 1
                    if 0 <= idx < len(available_docs):
                        selected.add(idx)
            except ValueError:
                logging.warning(f"Invalid selection part: {part}")
                continue
        
        return [available_docs[i] for i in sorted(selected)]
    
    def handle_existing_document(self, filename: str) -> bool:
        """
        Handle an existing document. Returns True if should process.
        """
        exists, status = self.check_document_exists(filename)
        if not exists:
            return True
            
        print(f"\nDocument '{filename}' already exists with status: {status}")
        choice = input("Options:\n1. Reprocess (delete and upload again)\n2. Skip\n3. Force update\nChoice (1-3): ")
        
        if choice == '1':
            if self.delete_document(filename):
                return True
            print("Failed to delete existing document, skipping...")
            return False
        elif choice == '3':
            return True
        else:  # choice 2 or invalid
            return False
    
    def batch_process_documents(self, available_docs: List[str]) -> List[str]:
        """
        Interactive document selection with advanced options.
        """
        while True:
            print("\nAvailable documents:")
            for idx, doc in enumerate(available_docs, 1):
                print(f"{idx}. {doc}")
            
            print("\nSelection options:")
            print("- Individual numbers: 1,3,5")
            print("- Ranges: 1-5")
            print("- Combinations: 1,3-5,7")
            print("- All documents: all")
            print("- Latest N documents: latest:N")
            print("- Press Enter to finish selection")
            
            selection = input("\nEnter your selection: ")
            if not selection:
                break
                
            selected = self.select_documents(available_docs, selection)
            if selected:
                return selected
            else:
                print("Invalid selection, please try again")
        
        return []

    def get_document_stats(self, filename: str) -> dict:
        """
        Get detailed statistics for a document.
        """
        c = self.conn.cursor()
        stats = {}
        
        # Get basic file info
        c.execute("SELECT status, chunk_count, processed_at FROM processed_files WHERE filename = ?", 
                 (filename,))
        result = c.fetchone()
        if result:
            stats.update({
                "status": result[0],
                "chunk_count": result[1],
                "processed_at": result[2]
            })
        
        # Get chunk stats
        c.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                SUM(CASE WHEN embedding_status = 'completed' THEN 1 ELSE 0 END) as completed_chunks,
                SUM(CASE WHEN embedding_status = 'failed' THEN 1 ELSE 0 END) as failed_chunks,
                AVG(token_count) as avg_tokens
            FROM chunks 
            WHERE filename = ?
        """, (filename,))
        chunk_stats = c.fetchone()
        if chunk_stats:
            stats.update({
                "total_chunks": chunk_stats[0],
                "completed_chunks": chunk_stats[1],
                "failed_chunks": chunk_stats[2],
                "avg_tokens_per_chunk": round(chunk_stats[3], 2) if chunk_stats[3] else 0
            })
        
        # Get Qdrant stats
        c.execute("""
            SELECT 
                COUNT(*) as total_vectors,
                SUM(CASE WHEN status = 'uploaded' THEN 1 ELSE 0 END) as uploaded_vectors
            FROM documents 
            WHERE filename = ?
        """, (filename,))
        vector_stats = c.fetchone()
        if vector_stats:
            stats.update({
                "total_vectors": vector_stats[0],
                "uploaded_vectors": vector_stats[1]
            })
        
        return stats
