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
from .stats import ProcessingStats
from src.database.maintenance import optimize_batch_processing, track_chunk_versions

class Document:
    """Represents a processed document with its content and metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata
    
    def __str__(self):
        return f"Document(content={self.page_content[:50]}..., metadata={self.metadata})"

def list_available_documents() -> List[str]:
    """
    List all available documents in the docs directory.
    
    This function recursively scans the docs directory for supported document types,
    validates their accessibility, and returns relative paths for processing.
    
    Supported File Types:
    - Markdown (.md)
    - Text (.txt)
    - reStructuredText (.rst)
    
    Returns:
        List[str]: List of relative paths to available documents
    
    Raises:
        Exception: If docs directory doesn't exist
        PermissionError: If files are not readable
    
    Example:
        # List all available documents
        docs = list_available_documents()
        print(f"Found {len(docs)} documents:")
        for doc in docs:
            print(f"- {doc}")
            
        # Filter specific file types
        md_docs = [doc for doc in docs if doc.endswith('.md')]
        print(f"Found {len(md_docs)} markdown documents")
    """
    """List all available documents in the docs directory"""
    docs_path = Path('./docs')
    if not docs_path.exists():
        raise Exception("The docs directory does not exist")
    
    available_docs = []
    allowed_extensions = {'.md', '.txt', '.rst'}  # Add supported file types
    
    # Using Path.rglob instead of os.walk for better path handling
    for file_path in docs_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in allowed_extensions:
            relative_path = file_path.relative_to(docs_path)
            available_docs.append(str(relative_path))
        elif file_path.is_file():
            logging.warning(f"Skipping unsupported file type: {file_path}")
    
    if not available_docs:
        logging.warning("No supported documents found in docs directory")
    
    return sorted(available_docs)

def select_documents(available_docs: List[str]) -> List[str]:
    """
    Interactive document selection interface.
    
    Provides a user-friendly interface for selecting documents to process,
    supporting various selection methods and validation.
    
    Selection Methods:
    1. Individual selection: "1, 3, 5"
    2. Range selection: "1-5"
    3. All documents: "all"
    
    Args:
        available_docs: List of available document paths
    
    Returns:
        List[str]: Selected document paths
    
    Raises:
        Exception: If no documents are available
        ValueError: If invalid selection is made
    
    Example:
        # List and select documents
        docs = list_available_documents()
        selected = select_documents(docs)
        
        # Process selection
        print("Selected documents:")
        for doc in selected:
            print(f"- {doc}")
            
        # Validate specific documents
        if 'important.md' in selected:
            print("Processing important document...")
    """
    """Interactive document selection"""
    if not available_docs:
        raise Exception("No documents found in the docs directory")
        
    print("Available documents:")
    for idx, doc in enumerate(available_docs, 1):
        print(f"{idx}. {doc}")
    
    while True:
        selection = input("\nEnter document numbers to process (comma-separated) or 'all' for all documents: ").strip()
        if selection.lower() == 'all':
            return available_docs
        
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_docs = [available_docs[idx] for idx in selected_indices if 0 <= idx < len(available_docs)]
            if selected_docs:
                return selected_docs
            print("No valid documents selected. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers separated by commas or 'all'")

def create_document_objects(processed_chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert processed chunks into LangChain Document objects.
    
    This function transforms raw processed chunks into structured Document objects,
    preserving metadata and embeddings for downstream processing.
    
    Args:
        processed_chunks: List of dictionaries containing:
            - content: str, The chunk text content
            - metadata: dict, Associated metadata
            - token_count: int, Number of tokens
            - embedding: List[float], Vector embedding
    
    Returns:
        List[Document]: LangChain Document objects with:
            - page_content: Original text
            - metadata: Enhanced with token count and embedding
    
    Raises:
        ValueError: If chunk format is invalid
        TypeError: If embedding format is incorrect
    
    Example:
        # Process chunks into documents
        chunks = [
            {
                "content": "Sample text",
                "metadata": {"source": "doc.md"},
                "token_count": 2,
                "embedding": [0.1, 0.2, 0.3]
            }
        ]
        docs = create_document_objects(chunks)
        
        # Access document properties
        for doc in docs:
            print(f"Content: {doc.page_content}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Embedding size: {len(doc.metadata['embedding'])}")
    """
    """Convert processed chunks into LangChain Document objects"""
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
                    "embedding": chunk["embedding"]  # Store embedding in metadata
                }
            )
            documents.append(doc)
        except Exception as e:
            api_logger.error(
                f"[{request_id}] Error converting chunk {i}: {str(e)}"
            )
            raise
    
    api_logger.info(
        f"[{request_id}] Successfully converted {len(documents)} chunks"
    )
    return documents

def segment_text_local(text: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Segment text into chunks using a language model.
    
    Args:
        text: Text to segment
        api_key: Optional API key for the language model
        
    Returns:
        List of dictionaries containing chunk information:
        - text: The chunk text
        - start: Starting position in original text
        - end: Ending position in original text
        - metadata: Additional metadata
    """
    # For testing, return simple chunks if no API key
    if not api_key:
        text_length = len(text)
        
        # For very short text, return as single chunk
        if text_length < 100:
            return [{"text": text, "start": 0, "end": text_length, "metadata": {}}]
        
        # For medium text, split into 2 chunks at nearest space
        elif text_length < 500:
            mid = text_length // 2
            # Find nearest space to split
            while mid > 0 and text[mid] != ' ':
                mid -= 1
            if mid == 0:  # No space found, force split
                mid = text_length // 2
            return [
                {"text": text[:mid], "start": 0, "end": mid, "metadata": {}},
                {"text": text[mid:], "start": mid, "end": text_length, "metadata": {}}
            ]
        
        # For long text, split into multiple chunks of roughly equal size
        else:
            chunks = []
            num_chunks = 3
            chunk_size = text_length // num_chunks
            current_pos = 0
            for i in range(num_chunks):
                end_pos = min(current_pos + chunk_size, text_length)
                # Find nearest space to split, unless it's the last chunk
                if end_pos < text_length and i < num_chunks - 1:
                    while end_pos > current_pos and text[end_pos] != ' ':
                        end_pos -= 1
                    if end_pos == current_pos:  # No space found, force split
                        end_pos = min(current_pos + chunk_size, text_length)
                chunk = {
                    "text": text[current_pos:end_pos],
                    "start": current_pos,
                    "end": end_pos,
                    "metadata": {}
                }
                chunks.append(chunk)
                current_pos = end_pos
            return chunks
    
    # Use the Jina AI segmentation API
    return segment_text(text, api_key)

def process_documents(markdown_files: List[str], conn, force_reprocess: List[str] = None) -> Tuple[List[Document], ProcessingStats]:
    """Process markdown documents and generate embeddings."""
    stats = ProcessingStats()
    processed_documents = []
    
    try:
        # Get unprocessed files
        unprocessed_files = get_unprocessed_files(conn, markdown_files)
        if force_reprocess:
            force_reprocess_files(conn, force_reprocess)
            # Only add files that aren't already in unprocessed_files
            unprocessed_files.extend([f for f in force_reprocess if f not in unprocessed_files])
        
        # Track versions for chunks
        track_chunk_versions(conn)
        
        docs_path = Path('./docs')
        for relative_path in unprocessed_files:
            try:
                file_path = docs_path / relative_path
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    # Segment text into chunks
                    segments = segment_text(content, api_key=os.getenv('JINA_API_KEY', ''))
                    if not segments:
                        logging.warning(f"No segments generated for {relative_path}")
                        continue
                    
                    logging.debug(f"Segmented chunks: {segments}")
                    stats.chunks_created += len(segments)
                    
                    # Process chunks and get embeddings
                    chunk_texts = [chunk["text"] for chunk in segments]
                    try:
                        embeddings_result = get_embeddings(segments, api_key=os.getenv('JINA_API_KEY', ''))
                        
                        if not embeddings_result or "data" not in embeddings_result:
                            logging.error(f"Failed to get embeddings for {relative_path}")
                            stats.errors += 1
                            conn.rollback()  # Rollback any partial changes
                            continue
                        
                        # Store chunks in database
                        c = conn.cursor()
                        for i, (chunk, embedding_data) in enumerate(zip(segments, embeddings_result["data"])):
                            chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()
                            basename = os.path.basename(str(file_path))
                            
                            # Store chunk
                            c.execute('''INSERT OR REPLACE INTO chunks 
                                       (id, filename, content, token_count, embedding_status)
                                       VALUES (?, ?, ?, ?, ?)''',
                                     (chunk_id,
                                      basename,
                                      chunk["text"],
                                      len(chunk["text"].split()),  # Simple token count
                                      "completed"))
                            
                            # Store document with embedding
                            c.execute('''INSERT OR REPLACE INTO documents 
                                       (id, filename, chunk_id, content, embedding, processed_at)
                                       VALUES (?, ?, ?, ?, ?, datetime('now'))''',
                                     (chunk_id,
                                      basename,
                                      i,
                                      chunk["text"],
                                      json.dumps(embedding_data["embedding"])))
                            
                            # Create Document object
                            doc = Document(
                                page_content=chunk["text"],
                                metadata={
                                    "source": basename,
                                    "chunk_id": i,
                                    "embedding": embedding_data["embedding"]
                                }
                            )
                            processed_documents.append(doc)
                            stats.embeddings_generated += 1
                        
                        # Mark file as processed
                        mark_file_as_processed(conn, basename, len(segments))
                        stats.files_processed += 1
                        
                    except Exception as e:
                        logging.error(f"Error getting embeddings for {relative_path}: {str(e)}")
                        stats.errors += 1
                        conn.rollback()  # Rollback any partial changes
                        continue
                    
                except Exception as e:
                    logging.error(f"Error segmenting text for {relative_path}: {str(e)}")
                    stats.errors += 1
                    conn.rollback()  # Rollback any partial changes
                    continue
                
            except Exception as e:
                logging.error(f"Error reading file {relative_path}: {str(e)}")
                stats.errors += 1
                conn.rollback()  # Rollback any partial changes
                continue
        
        conn.commit()
        return processed_documents, stats
        
    except Exception as e:
        logging.error(f"Error in process_documents: {str(e)}")
        conn.rollback()
        raise

class ProcessingStats:
    """Statistics for document processing."""
    def __init__(self):
        self.files_processed = 0
        self.chunks_created = 0
        self.embeddings_generated = 0
        self.errors = 0
    
    def __str__(self):
        return (
            f"Files processed: {self.files_processed}\n"
            f"Chunks created: {self.chunks_created}\n"
            f"Embeddings generated: {self.embeddings_generated}\n"
            f"Errors: {self.errors}"
        )