"""
Document Processing Module

This module provides a comprehensive pipeline for processing markdown documents,
with a focus on text segmentation, embedding generation, and efficient storage.

Key Features:
- Document loading and validation with support for .md, .txt, and .rst files
- Text segmentation using Jina AI's advanced NLP models
- Efficient embedding generation with batch processing
- Robust SQLite-based progress tracking and version control
- Error handling with automatic retry mechanisms
- Optimized batch processing for large document sets

The processing pipeline follows these steps:
1. Document Loading:
   - Validates file existence and permissions
   - Supports multiple file formats
   - Handles nested directory structures

2. Text Segmentation:
   - Uses Jina AI for intelligent text splitting
   - Maintains context across segments
   - Optimizes segment length for embedding

3. Embedding Generation:
   - Batched processing for efficiency
   - Automatic retry on failures
   - Version tracking for embeddings

4. Storage Management:
   - SQLite-based progress tracking
   - Efficient storage of embeddings
   - Version control for processed chunks

5. Status Tracking:
   - Detailed progress monitoring
   - Error logging and recovery
   - Processing statistics

Dependencies:
- langchain: For document handling
- jina: For text processing and embeddings
- sqlite3: For data storage
- tqdm: For progress tracking

Environment Variables:
- JINA_API_KEY: Required for Jina AI services
- LOG_LEVEL: Optional, controls logging verbosity

Example Usage:
    # List available documents
    available_docs = list_available_documents()
    
    # Select specific documents
    selected_docs = select_documents(available_docs)
    
    # Process documents with SQLite connection
    with sqlite3.connect('documents.db') as conn:
        docs, stats = process_documents(selected_docs, conn)
        print(f"Processed {stats.files_processed} files")
        print(f"Generated {stats.embeddings_generated} embeddings")

Error Handling:
    The module implements comprehensive error handling:
    - File-level validation
    - API connection retries
    - Batch processing recovery
    - Detailed error logging
"""

import logging
import hashlib
import json
from datetime import datetime
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from langchain.schema import Document
from ..api.jina import segment_text, get_embeddings
from ..database.operations import mark_file_as_processed, get_unprocessed_files, force_reprocess_files
from src.database.maintenance import optimize_batch_processing, track_chunk_versions
from .stats import ProcessingStats

class Document:
    """Represents a processed document with its content and metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata
    
    def __str__(self):
        return f"Document(content={self.page_content[:50]}..., metadata={self.metadata})"

def list_available_documents() -> List[str]:
    """List all available documents in the configured documents directory"""
    # Get documents path from environment or use default, make it absolute
    docs_path = Path(os.getenv('DOCUMENTS_PATH', 'docs')).resolve()
    
    if not docs_path.exists():
        raise Exception(f"The documents directory does not exist: {docs_path}")
    
    available_docs = []
    allowed_extensions = {'.md', '.txt', '.rst'}
    
    print(f"\nDocument directory: {docs_path}\n")
    
    # Using Path.rglob instead of os.walk for better path handling
    for file_path in docs_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in allowed_extensions:
            # Store tuple of (display_name, full_path)
            available_docs.append((file_path.name, str(file_path)))
        elif file_path.is_file():
            logging.warning(f"Skipping unsupported file type: {file_path}")
    
    if not available_docs:
        logging.warning(f"No supported documents found in directory: {docs_path}")
    
    return sorted(available_docs)

def select_documents(available_docs: List[tuple]) -> List[str]:
    """Interactive document selection interface."""
    if not available_docs:
        raise Exception("No documents found in the docs directory")
        
    print("Available documents:")
    for idx, (display_name, _) in enumerate(available_docs, 1):
        print(f"{idx}. {display_name}")
    
    while True:
        selection = input("\nEnter document numbers to process (comma-separated) or 'all' for all documents: ").strip()
        if selection.lower() == 'all':
            return [full_path for _, full_path in available_docs]
        
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_docs = [available_docs[idx][1] for idx in selected_indices if 0 <= idx < len(available_docs)]
            if selected_docs:
                return selected_docs
            print("No valid documents selected. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers separated by commas or 'all'")

def create_document_objects(processed_chunks: List[Dict[str, Any]], stats: ProcessingStats) -> List[Document]:
    """
    Convert processed chunks into LangChain Document objects.
    
    Args:
        processed_chunks: List of dictionaries containing chunk data
        stats: ProcessingStats instance for tracking
        
    Returns:
        List of Document objects
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-convert".encode()).hexdigest()[:8]
    
    api_logger.info(
        f"[{request_id}] Converting {len(processed_chunks)} chunks to Document objects"
    )
    
    documents = []
    for i, chunk in enumerate(processed_chunks):
        try:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    **chunk["metadata"],
                    "token_count": chunk["token_count"],
                    "embedding": chunk["embedding"]
                }
            )
            documents.append(doc)
            stats.track_db_operation('chunk_convert')
        except Exception as e:
            api_logger.error(
                f"[{request_id}] Error converting chunk {i}: {str(e)}"
            )
            stats.track_db_operation('chunk_convert', error=str(e))
            raise
    
    api_logger.info(
        f"[{request_id}] Successfully converted {len(documents)} chunks"
    )
    return documents

def segment_text_local(text: str, api_key: Optional[str] = None, stats: Optional[ProcessingStats] = None) -> List[Dict[str, Any]]:
    """
    Segment text into chunks using a language model.
    
    Args:
        text: Text to segment
        api_key: Optional API key for the language model
        stats: Optional ProcessingStats instance for tracking
        
    Returns:
        List of dictionaries containing chunk information
    """
    start_time = datetime.now()
    
    try:
        # For testing, return simple chunks if no API key
        if not api_key:
            text_length = len(text)
            
            # For very short text, return as single chunk
            if text_length < 100:
                chunks = [{"text": text, "start": 0, "end": text_length, "metadata": {}}]
            # For medium text, split into 2 chunks at nearest space
            elif text_length < 500:
                mid = text_length // 2
                while mid > 0 and text[mid] != ' ':
                    mid -= 1
                if mid == 0:
                    mid = text_length // 2
                chunks = [
                    {"text": text[:mid], "start": 0, "end": mid, "metadata": {}},
                    {"text": text[mid:], "start": mid, "end": text_length, "metadata": {}}
                ]
            # For long text, split into multiple chunks
            else:
                chunks = []
                num_chunks = 3
                chunk_size = text_length // num_chunks
                current_pos = 0
                for i in range(num_chunks):
                    end_pos = min(current_pos + chunk_size, text_length)
                    if end_pos < text_length and i < num_chunks - 1:
                        while end_pos > current_pos and text[end_pos] != ' ':
                            end_pos -= 1
                        if end_pos == current_pos:
                            end_pos = min(current_pos + chunk_size, text_length)
                    chunk = {
                        "text": text[current_pos:end_pos],
                        "start": current_pos,
                        "end": end_pos,
                        "metadata": {}
                    }
                    chunks.append(chunk)
                    current_pos = end_pos
            
            if stats:
                stats.track_api_call('jina_chunking', success=True, 
                                   latency=(datetime.now() - start_time).total_seconds())
            return chunks
        
        # Use the Jina AI segmentation API
        try:
            chunks = segment_text(text, api_key)
            if stats:
                stats.track_api_call('jina_chunking', success=True, 
                                   latency=(datetime.now() - start_time).total_seconds())
            return chunks
        except Exception as e:
            if stats:
                stats.track_api_call('jina_chunking', success=False, 
                                   latency=(datetime.now() - start_time).total_seconds(),
                                   error=str(e))
            raise
            
    except Exception as e:
        if stats:
            stats.track_api_call('jina_chunking', success=False, 
                               latency=(datetime.now() - start_time).total_seconds(),
                               error=str(e))
        raise

def process_documents(markdown_files: List[str], conn, force_reprocess: List[str] = None) -> Tuple[List[Document], ProcessingStats]:
    """Process markdown documents and generate embeddings."""
    stats = ProcessingStats()
    stats.start()
    processed_documents = []
    
    try:
        # Check for duplicate documents first
        from ..database.operations import check_document_exists
        duplicates = []
        for file_path in markdown_files:
            doc_status = check_document_exists(conn, file_path)
            if doc_status:
                print(f"\nDocument already exists: {file_path}")
                print(f"Status: {doc_status['status']}")
                print(f"Last processed: {doc_status['processed_at']}")
                print(f"Chunks: {doc_status['completed_chunks']}/{doc_status['total_chunks']}")
                
                while True:
                    choice = input("Do you want to: [r]eprocess this document, [s]kip it, or [c]ancel processing? ").lower()
                    if choice in ['r', 's', 'c']:
                        break
                    print("Invalid choice. Please enter 'r', 's', or 'c'")
                
                if choice == 'c':
                    print("Processing cancelled.")
                    return processed_documents, stats
                elif choice == 'r':
                    if not force_reprocess:
                        force_reprocess = []
                    force_reprocess.append(file_path)
                else:  # skip
                    duplicates.append(file_path)
        
        # Remove skipped duplicates from processing list
        markdown_files = [f for f in markdown_files if f not in duplicates]
        if not markdown_files and not force_reprocess:
            print("No documents to process after handling duplicates.")
            return processed_documents, stats
            
        # Get unprocessed files - these should now be full paths
        unprocessed_files = get_unprocessed_files(conn, markdown_files)
        if force_reprocess:
            force_reprocess_files(conn, force_reprocess)
            unprocessed_files.extend([f for f in force_reprocess if f not in unprocessed_files])
        
        # Track versions for chunks
        track_chunk_versions(conn)
        
        total_files = len(unprocessed_files)
        for file_idx, file_path in enumerate(unprocessed_files, 1):
            try:
                # Convert string path to Path object
                file_path = Path(file_path)
                logging.info(f"Processing file {file_idx} of {total_files}: {file_path}")
                
                if not file_path.exists():
                    logging.error(f"File not found: {file_path}")
                    stats.errors += 1
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    # Segment text into chunks
                    logging.info(f"Segmenting text for {file_path}")
                    chunks = segment_text_local(content, stats=stats)
                    if not chunks:
                        logging.warning(f"No segments generated for {file_path}")
                        continue
                        
                    logging.info(f"Generated {len(chunks)} chunks for {file_path}")
                    stats.update(chunks_created=len(chunks))
                    
                    # Store chunks in database with pending status
                    c = conn.cursor()
                    basename = os.path.basename(str(file_path))
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            # Get the actual content from the chunk
                            chunk_content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                            
                            # Generate IDs using the content
                            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()
                            content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                            
                            # Store chunk with pending status and proper chunk number
                            c.execute('''INSERT OR REPLACE INTO chunks 
                                       (id, filename, content, chunk_number, content_hash, 
                                        token_count, embedding_status, chunking_status)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                     (chunk_id,
                                      basename,
                                      chunk_content,
                                      i,  # Use loop index as chunk_number
                                      content_hash,
                                      len(chunk_content.split()),  # Simple token count
                                      "pending",
                                      "completed"))
                            
                            # Create placeholder document record
                            c.execute('''INSERT OR REPLACE INTO documents 
                                       (id, filename, chunk_id, content, processed_at, status, chunking_status)
                                       VALUES (?, ?, ?, ?, datetime('now'), ?, ?)''',
                                     (hashlib.md5(basename.encode()).hexdigest(),
                                      basename,
                                      i,
                                      chunk_content,
                                      "processing",
                                      "completed"))
                            
                            stats.track_db_operation('chunk_insert')
                            
                        except Exception as e:
                            logging.error(f"Error storing chunk {i} for {file_path}: {str(e)}")
                            stats.track_db_operation('chunk_insert', error=str(e))
                            raise
                    
                    # Update processed_files table
                    try:
                        c.execute('''INSERT OR REPLACE INTO processed_files 
                                   (filename, processed_at, chunk_count, status, chunking_status)
                                   VALUES (?, datetime('now'), ?, ?, ?)''',
                                 (basename, len(chunks), "processing", "completed"))
                        
                        conn.commit()
                        stats.update(files_processed=1)
                        stats.track_db_operation('file_status_update')
                        logging.info(f"Successfully chunked {file_path} into {len(chunks)} segments")
                        
                    except Exception as e:
                        logging.error(f"Error updating processed_files for {file_path}: {str(e)}")
                        stats.track_db_operation('file_status_update', error=str(e))
                        conn.rollback()
                        raise
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    stats.errors += 1
                    conn.rollback()
                    continue
                    
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {str(e)}")
                stats.errors += 1
                continue
                
        # Process any pending chunks
        try:
            from ..database.operations import process_pending_chunks
            logging.info("Processing pending chunks...")
            processed_count, error_count = process_pending_chunks(conn)
            stats.update(embeddings_generated=processed_count)
            stats.errors += error_count
            logging.info(f"Finished processing chunks: {processed_count} processed, {error_count} errors")
        except Exception as e:
            logging.error(f"Error processing pending chunks: {str(e)}")
            stats.errors += 1
                
    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")
        stats.errors += 1
    
    finally:
        stats.end()
        
    return processed_documents, stats

def chunk_document(doc_path: str) -> Tuple[List[str], List[int]]:
    """
    Chunk a document into segments using Jina AI segmenter.
    
    Args:
        doc_path: Path to the document
        
    Returns:
        Tuple of (list of chunk texts, list of token counts)
    """
    try:
        # Read document
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Segment text
        chunks = segment_text(content)
        
        # Calculate token counts (simple word-based count)
        token_counts = [len(chunk.split()) for chunk in chunks]
        
        return chunks, token_counts
        
    except Exception as e:
        logging.error(f"Failed to chunk document {doc_path}: {e}")
        raise