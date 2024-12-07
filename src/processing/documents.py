import logging
import hashlib
import json
from datetime import datetime
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from langchain.schema import Document
from ..api.jina import segment_text, get_embeddings
from ..database.operations import mark_file_as_processed, get_unprocessed_files, force_reprocess_files
from .stats import ProcessingStats
from src.database.maintenance import optimize_batch_processing, track_chunk_versions

def list_available_documents():
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

def select_documents(available_docs):
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

def process_documents(markdown_files: List[str], conn, 
                     force_reprocess: List[str] = None) -> Tuple[List[Document], ProcessingStats]:
    """Process documents through segmentation and embedding"""
    stats = {"files_processed": 0, "chunks_created": 0, "embeddings_generated": 0}
    processed_documents = []
    
    # Ensure chunk versioning is set up
    track_chunk_versions(conn)
    
    for file in markdown_files:
        try:
            # Check if we need to segment the file
            c = conn.cursor()
            c.execute("SELECT status, chunk_count FROM processed_files WHERE filename = ?", (file,))
            file_status = c.fetchone()
            
            if not file_status or force_reprocess or file_status[0] == 'failed':
                # Segment the document
                chunks = segment_text(file)
                stats["chunks_created"] += len(chunks)
                
                # Store chunks in database
                now = datetime.now()
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file}_{i}"
                    c.execute("""INSERT OR REPLACE INTO chunks 
                                (id, filename, chunk_number, content, token_count, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)""",
                            (chunk_id, file, i, chunk, len(chunk.split()), now))
                
                # Update file status
                c.execute("""INSERT OR REPLACE INTO processed_files 
                            (filename, last_modified, processed_at, chunk_count, status)
                            VALUES (?, ?, ?, ?, 'segmented')""",
                         (file, os.path.getmtime(file), now, len(chunks)))
                conn.commit()
            
            # Process chunks that need embeddings in optimized batches
            c.execute("""SELECT id, content FROM chunks 
                        WHERE filename = ? AND embedding_status = 'pending'""", (file,))
            pending_chunks = c.fetchall()
            
            if pending_chunks:
                chunk_contents = [content for _, content in pending_chunks]
                chunk_ids = [chunk_id for chunk_id, _ in pending_chunks]
                
                # Process in optimized batches
                for batch_idx, (content_batch, batch_stats) in enumerate(optimize_batch_processing(chunk_contents)):
                    try:
                        # Get embeddings for batch
                        embeddings = get_embeddings(content_batch)
                        stats["embeddings_generated"] += len(embeddings)
                        
                        # Store embeddings for batch
                        for content, embedding, chunk_id in zip(content_batch, embeddings, 
                                                              chunk_ids[batch_idx*50:(batch_idx+1)*50]):
                            # Store in documents table
                            doc_id = f"doc_{chunk_id}"
                            c.execute("""INSERT OR REPLACE INTO documents 
                                        (id, chunk_id, filename, content, embedding_id, embedding, processed_at)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                    (doc_id, chunk_id, file, content, 
                                     str(hashlib.md5(content.encode()).hexdigest()), 
                                     json.dumps(embedding), datetime.now()))
                            
                            # Update chunk status
                            c.execute("""UPDATE chunks 
                                       SET embedding_status = 'completed', 
                                           processed_at = ? 
                                       WHERE id = ?""",
                                    (datetime.now(), chunk_id))
                            
                            # Add to processed documents
                            processed_documents.append(Document(
                                page_content=content,
                                metadata={"source": file, "embedding": embedding}
                            ))
                        
                        conn.commit()
                        logging.info(f"Processed batch {batch_idx + 1} with {len(content_batch)} chunks")
                        
                    except Exception as e:
                        logging.error(f"Failed to process batch {batch_idx + 1}: {str(e)}")
                        # Mark chunks as failed
                        for chunk_id in chunk_ids[batch_idx*50:(batch_idx+1)*50]:
                            c.execute("UPDATE chunks SET embedding_status = 'failed' WHERE id = ?", 
                                    (chunk_id,))
                        conn.commit()
                
                # Update file status if all chunks are processed
                c.execute("""UPDATE processed_files SET status = 'embedded' 
                            WHERE filename = ? AND 
                            (SELECT COUNT(*) FROM chunks 
                             WHERE filename = ? AND embedding_status != 'completed') = 0""",
                         (file, file))
                conn.commit()
            
            stats["files_processed"] += 1
            
        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")
            continue
    
    return processed_documents, stats