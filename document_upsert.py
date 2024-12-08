import os
import sqlite3
import logging
import time
import requests
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import JinaEmbeddings 
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from dotenv import load_dotenv
import hashlib
from datetime import datetime
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def setup_logging():
    """Configure detailed logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    # Add specific loggers for APIs
    api_logger = logging.getLogger('api_calls')
    api_logger.setLevel(logging.INFO)
    api_handler = logging.FileHandler('api_calls.log')
    api_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    api_logger.addHandler(api_handler)
    return api_logger

def check_environment():
    """Validate required environment variables and directories"""
    required_vars = ["JINA_API_KEY", "QDRANT_API_KEY", "QDRANT_URL", "QDRANT_COLLECTION_NAME"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Check docs directory
    docs_path = Path('./docs')
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logging.info("Created docs directory")
    
    return True

def initialize_database():
    """Initialize SQLite database with proper schema and indices"""
    logging.info("Setting up SQLite database...")
    
    # Create docs directory if it doesn't exist
    docs_path = Path('./docs')
    if not docs_path.exists():
        docs_path.mkdir(parents=True)
        logging.info("Created docs directory")
    
    conn = sqlite3.connect('records.db')
    c = conn.cursor()
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON")
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                (id TEXT PRIMARY KEY,
                 filename TEXT,
                 chunk_id INTEGER,
                 content TEXT,
                 embedding_id TEXT,
                 processed_at TIMESTAMP,
                 UNIQUE(filename, chunk_id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS processed_files
                (filename TEXT PRIMARY KEY,
                 last_modified TIMESTAMP,
                 processed_at TIMESTAMP)''')
    
    # Create indices for better query performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_processed ON processed_files(processed_at)')
    
    conn.commit()
    return conn

def cleanup_database(conn: sqlite3.Connection, days_threshold: int = 30) -> None:
    """Clean up old and orphaned records from the database"""
    c = conn.cursor()
    logging.info(f"Starting database cleanup (threshold: {days_threshold} days)...")
    
    try:
        # Remove records for files that no longer exist
        c.execute('''SELECT DISTINCT filename FROM documents''')
        all_files = c.fetchall()
        docs_path = Path('./docs')
        
        for (filename,) in all_files:
            if not (docs_path / filename).exists():
                logging.info(f"Removing records for deleted file: {filename}")
                c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
                c.execute('DELETE FROM processed_files WHERE filename = ?', (filename,))
        
        # Remove old records based on threshold
        c.execute(f'''DELETE FROM documents 
                     WHERE processed_at < datetime('now', '-{days_threshold} days')''')
        old_docs_count = c.rowcount
        
        c.execute(f'''DELETE FROM processed_files 
                     WHERE processed_at < datetime('now', '-{days_threshold} days')
                     AND filename NOT IN (SELECT DISTINCT filename FROM documents)''')
        old_files_count = c.rowcount
        
        conn.commit()
        logging.info(f"Cleanup complete. Removed {old_docs_count} old document chunks and {old_files_count} old file records")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error during database cleanup: {str(e)}")
        raise

def get_database_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get statistics about the database contents"""
    c = conn.cursor()
    stats = {}
    
    try:
        c.execute('SELECT COUNT(DISTINCT filename) FROM documents')
        stats['total_files'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM documents')
        stats['total_chunks'] = c.fetchone()[0]
        
        c.execute('''SELECT filename, processed_at 
                    FROM processed_files 
                    ORDER BY processed_at DESC 
                    LIMIT 5''')
        stats['recent_files'] = c.fetchall()
        
        c.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['database_size'] = c.fetchone()[0]
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting database stats: {str(e)}")
        raise

def get_unprocessed_files(available_docs: List[str], conn: sqlite3.Connection) -> List[str]:
    """Filter out already processed files that haven't been modified"""
    c = conn.cursor()
    unprocessed_files = []
    
    for doc in available_docs:
        file_path = Path('./docs') / doc
        last_modified = file_path.stat().st_mtime
        
        # Check if file has been processed and not modified since
        c.execute('''SELECT last_modified FROM processed_files 
                    WHERE filename = ?''', (str(doc),))
        result = c.fetchone()
        
        if not result or result[0] < last_modified:
            unprocessed_files.append(doc)
    
    return unprocessed_files

def mark_file_as_processed(filename: str, conn: sqlite3.Connection):
    """Mark a file as processed with its current modification time"""
    c = conn.cursor()
    file_path = Path('./docs') / filename
    last_modified = file_path.stat().st_mtime
    
    c.execute('''INSERT OR REPLACE INTO processed_files 
                (filename, last_modified, processed_at)
                VALUES (?, ?, datetime('now'))''',
             (str(filename), last_modified))
    conn.commit()

def segment_text(text: str, api_key: str) -> Dict[str, Any]:
    """Segment text using Jina AI's Segmenter API"""
    api_logger = logging.getLogger('api_calls')
    url = "https://segment.jina.ai/"
    
    # Log request details
    request_id = hashlib.md5(f"{datetime.now()}-segment".encode()).hexdigest()[:8]
    api_logger.info(f"[{request_id}] Segmenter API Request - Content length: {len(text)} chars")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "content": text,
        "tokenizer": "o200k_base",
        "return_tokens": True,
        "return_chunks": True,
        "max_chunk_length": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Log response details
        api_logger.info(
            f"[{request_id}] Segmenter API Response - "
            f"Chunks: {len(result.get('chunks', []))} | "
            f"Tokens: {result.get('num_tokens', 0)}"
        )
        return result
        
    except Exception as e:
        api_logger.error(f"[{request_id}] Segmenter API Error: {str(e)}")
        raise

def get_embeddings(texts: List[str], api_key: str, batch_size: int = 100) -> List[List[float]]:
    """Get embeddings using Jina AI's Embeddings API with batching"""
    api_logger = logging.getLogger('api_calls')
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        request_id = hashlib.md5(f"{datetime.now()}-embed-{i}".encode()).hexdigest()[:8]
        
        api_logger.info(
            f"[{request_id}] Embeddings API Request - "
            f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} | "
            f"Size: {len(batch)} texts"
        )
        
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "jina-embeddings-v3",
            "input": batch,
            "task": "retrieval.passage",
            "dimensions": 1024,
            "late_chunking": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)
            
            api_logger.info(
                f"[{request_id}] Embeddings API Response - "
                f"Successfully embedded {len(batch_embeddings)} texts"
            )
            
        except Exception as e:
            api_logger.error(f"[{request_id}] Embeddings API Error: {str(e)}")
            raise
    
    return all_embeddings

class ProcessingStats:
    """Track document processing statistics"""
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.total_chunks = 0
        self.new_chunks = 0
        self.start_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'skipped_files': self.skipped_files,
            'failed_files': self.failed_files,
            'total_chunks': self.total_chunks,
            'new_chunks': self.new_chunks,
            'elapsed_time': elapsed_time,
            'processing_rate': self.processed_files / elapsed_time if elapsed_time > 0 else 0,
            'chunk_rate': self.total_chunks / elapsed_time if elapsed_time > 0 else 0
        }
    
    def log_summary(self):
        summary = self.get_summary()
        logging.info("Processing Summary:")
        logging.info(f"Files: {summary['processed_files']} processed, {summary['skipped_files']} skipped, {summary['failed_files']} failed")
        logging.info(f"Chunks: {summary['new_chunks']} new out of {summary['total_chunks']} total")
        logging.info(f"Time elapsed: {summary['elapsed_time']:.2f} seconds")
        logging.info(f"Processing rate: {summary['processing_rate']:.2f} files/second")
        logging.info(f"Chunk processing rate: {summary['chunk_rate']:.2f} chunks/second")

def list_available_documents():
    docs_path = Path('./docs')
    if not docs_path.exists():
        raise Exception("The docs directory does not exist")
    
    available_docs = []
    allowed_extensions = {'.md', '.txt', '.rst'}  # Add supported file types
    
    for root, _, files in os.walk(docs_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in allowed_extensions:
                relative_path = file_path.relative_to(docs_path)
                available_docs.append(str(relative_path))
            else:
                logging.warning(f"Skipping unsupported file type: {file}")
    
    if not available_docs:
        logging.warning("No supported documents found in docs directory")
    
    return sorted(available_docs)

def select_documents(available_docs):
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

def process_markdown_document(content: str, jina_api_key: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Process a markdown document with retry logic for API rate limits"""
    for attempt in range(max_retries):
        try:
            segments = segment_text(content, jina_api_key)
            if not segments.get("chunks"):
                logging.warning("No chunks were generated from the segmenter")
                return []
                
            chunk_embeddings = get_embeddings(segments["chunks"], jina_api_key)
            
            processed_chunks = []
            for chunk, embedding in zip(segments["chunks"], chunk_embeddings):
                processed_chunks.append({
                    "content": chunk,
                    "embedding": embedding,
                    "token_count": segments["num_tokens"] // len(segments["chunks"]),
                })
                
            return processed_chunks
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit exceeded
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff
                    logging.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            logging.error(f"Error processing document: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise

def process_documents(markdown_files: List[str], conn: sqlite3.Connection, 
                     force_reprocess: List[str] = None) -> Tuple[List[Document], ProcessingStats]:
    """Process a list of markdown files using Jina AI APIs"""
    processed_documents = []
    stats = ProcessingStats()
    jina_api_key = os.environ["JINA_API_KEY"]
    api_logger = logging.getLogger('api_calls')
    
    # Log processing start
    request_id = hashlib.md5(f"{datetime.now()}-process".encode()).hexdigest()[:8]
    
    # Test document loading first
    api_logger.info(f"[{request_id}] Testing document loading...")
    test_document_loading('./docs', "**/*.md")
    
    # Test Jina APIs
    api_logger.info(f"[{request_id}] Testing Jina APIs...")
    test_jina_apis()
    
    api_logger.info(
        f"[{request_id}] Starting document processing - "
        f"Files to process: {len(markdown_files)}"
    )
    
    # Force reprocess if requested
    if force_reprocess:
        force_reprocess_files(force_reprocess, conn)
        api_logger.info(
            f"[{request_id}] Force reprocessing {len(force_reprocess)} files"
        )
    
    # Get unprocessed files
    unprocessed_files = get_unprocessed_files(markdown_files, conn)
    stats.total_files = len(markdown_files)
    stats.skipped_files = stats.total_files - len(unprocessed_files)
    
    api_logger.info(
        f"[{request_id}] Found {len(unprocessed_files)} files to process, "
        f"{stats.skipped_files} files skipped"
    )
    
    if not unprocessed_files:
        api_logger.info(f"[{request_id}] All files are up to date, no processing needed")
        return [], stats
    
    # Load documents using DirectoryLoader
    loader = DirectoryLoader('./docs', glob="**/*.md")
    docs = loader.load()
    
    api_logger.info(f"[{request_id}] Loaded {len(docs)} documents for processing")
    
    c = conn.cursor()
    
    # Process each document with progress bar
    for doc in tqdm(docs, desc="Processing documents"):
        relative_path = Path(doc.metadata['source']).relative_to(Path('./docs'))
        api_logger.info(f"Processing document: {relative_path}")
        
        try:
            with open(doc.metadata['source'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use Jina Segmenter API to chunk the document
            segments = segment_text(content, jina_api_key)
            if not segments.get("chunks"):
                logging.warning(f"No chunks generated for {relative_path}")
                stats.failed_files += 1
                continue
                
            # Get embeddings for all chunks
            chunk_embeddings = get_embeddings(segments["chunks"], jina_api_key)
            stats.total_chunks += len(segments["chunks"])
            
            # Process chunks with progress bar
            for i, (chunk, embedding) in enumerate(tqdm(
                zip(segments["chunks"], chunk_embeddings), 
                desc=f"Processing chunks for {relative_path}",
                total=len(segments["chunks"]),
                leave=False
            )):
                chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                
                # Check if chunk already exists and is unchanged
                c.execute('''SELECT id FROM documents 
                           WHERE id = ? AND content = ?''',
                        (chunk_id, chunk))
                
                if not c.fetchone():
                    stats.new_chunks += 1
                    processed_chunk = {
                        "content": chunk,
                        "embedding": embedding,
                        "token_count": segments["num_tokens"] // len(segments["chunks"]),
                        "metadata": {
                            "source": str(relative_path),
                            "chunk_size": len(chunk),
                            "chunk_id": i
                        }
                    }
                    processed_documents.append(processed_chunk)
                    
                    # Store in database
                    c.execute('''INSERT OR REPLACE INTO documents 
                               (id, filename, chunk_id, content, embedding_id, processed_at)
                               VALUES (?, ?, ?, ?, ?, datetime('now'))''',
                             (chunk_id,
                              str(relative_path),
                              i,
                              chunk,
                              str(embedding[:5])))  # Store first 5 values as reference
            
            # Mark file as processed
            mark_file_as_processed(str(relative_path), conn)
            conn.commit()
            stats.processed_files += 1
            
            api_logger.info(
                f"Successfully processed {relative_path}: "
                f"{len(segments['chunks'])} chunks, "
                f"{stats.new_chunks} new"
            )
            
        except Exception as e:
            error_msg = f"Error processing file {relative_path}: {str(e)}"
            api_logger.error(error_msg)
            logging.error(error_msg)
            stats.failed_files += 1
            continue
    
    # Convert to LangChain Document objects before returning
    langchain_documents = create_document_objects(processed_documents)
    api_logger.info(
        f"[{request_id}] Processing complete - "
        f"Created {len(langchain_documents)} LangChain documents"
    )
    return langchain_documents, stats

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

def initialize_qdrant(documents: List[Document], max_retries: int = 3):
    """Initialize Qdrant with retry logic and verification"""
    api_logger = logging.getLogger('api_calls')
    request_id = log_qdrant_operation("Initialization", {
        "document_count": len(documents),
        "collection": os.environ['QDRANT_COLLECTION_NAME']
    })
    
    embeddings = JinaEmbeddings(
        jina_api_key=os.environ["JINA_API_KEY"],
        model_name="jina-embeddings-v3"
    )
    
    for attempt in range(max_retries):
        try:
            qdrant = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                url=os.environ["QDRANT_URL"],
                collection_name=os.environ["QDRANT_COLLECTION_NAME"],
                prefer_grpc=True,
                api_key=os.environ["QDRANT_API_KEY"]
            )
            
            # Verify upload by checking a sample of documents
            verification_size = min(5, len(documents))
            sample_docs = documents[:verification_size]
            
            api_logger.info(
                f"[{request_id}] Verifying upload with {verification_size} sample documents"
            )
            
            for idx, doc in enumerate(sample_docs):
                results = qdrant.similarity_search(doc.page_content, k=1)
                if not results or results[0].page_content != doc.page_content:
                    error_details = {
                        "error": "Verification failed",
                        "sample_index": idx,
                        "expected": doc.page_content[:100] + "...",
                        "received": results[0].page_content[:100] + "..." if results else "No results"
                    }
                    log_qdrant_operation("Verification", error_details, success=False)
                    raise Exception("Document verification failed - content mismatch")
                
                api_logger.info(
                    f"[{request_id}] Verified sample document {idx + 1}/{verification_size}"
                )
            
            success_details = {
                "documents_uploaded": len(documents),
                "samples_verified": verification_size,
                "collection": os.environ['QDRANT_COLLECTION_NAME']
            }
            log_qdrant_operation("Verification", success_details)
            
            logging.info(
                f"Qdrant initialization complete and verified - "
                f"Successfully uploaded {len(documents)} documents"
            )
            return qdrant
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                error_details = {
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "wait_time": wait_time,
                    "error": str(e)
                }
                log_qdrant_operation("Retry", error_details, success=False)
                time.sleep(wait_time)
                continue
                
            error_details = {
                "final_attempt": attempt + 1,
                "error": str(e)
            }
            log_qdrant_operation("Failed", error_details, success=False)
            raise

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

def force_reprocess_files(filenames: List[str], conn: sqlite3.Connection) -> None:
    """
    Force reprocessing of specific files by removing their records
    
    Args:
        filenames: List of filenames to reprocess (relative to docs directory)
        conn: SQLite connection
    """
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-force-reprocess".encode()).hexdigest()[:8]
    
    api_logger.info(
        f"[{request_id}] Force reprocessing initiated for {len(filenames)} files"
    )
    
    c = conn.cursor()
    
    try:
        for filename in filenames:
            # Delete existing records
            c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
            c.execute('DELETE FROM processed_files WHERE filename = ?', (filename,))
            
            api_logger.info(
                f"[{request_id}] Cleared records for {filename}"
            )
        
        conn.commit()
        api_logger.info(
            f"[{request_id}] Successfully marked {len(filenames)} files for reprocessing"
        )
        
    except Exception as e:
        conn.rollback()
        error_details = {
            "error": str(e),
            "files": filenames
        }
        api_logger.error(
            f"[{request_id}] Failed to force reprocess files: {json.dumps(error_details)}"
        )
        raise Exception(f"Error marking files for reprocessing: {str(e)}")

def test_document_loading(docs_path: str, glob_pattern: str) -> None:
    """Test document loading functionality"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-loading".encode()).hexdigest()[:8]
    
    try:
        # Test direct file existence
        full_path = Path(docs_path)
        api_logger.info(f"[{request_id}] Testing path: {full_path.absolute()}")
        api_logger.info(f"[{request_id}] Path exists: {full_path.exists()}")
        api_logger.info(f"[{request_id}] Files in directory: {list(full_path.glob('*'))}")
        
        # Test DirectoryLoader
        loader = DirectoryLoader(docs_path, glob=glob_pattern)
        docs = loader.load()
        api_logger.info(
            f"[{request_id}] DirectoryLoader found {len(docs)} documents "
            f"using pattern: {glob_pattern}"
        )
        
        # Print details of each found document
        for doc in docs:
            api_logger.info(
                f"[{request_id}] Found document: {doc.metadata['source']}, "
                f"Size: {len(doc.page_content)} chars"
            )
            
    except Exception as e:
        api_logger.error(f"[{request_id}] Document loading test failed: {str(e)}")
        raise

def test_jina_apis(test_text: str = "This is a test document. Let's see how it gets processed."):
    """Test Jina API functionality"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-apis".encode()).hexdigest()[:8]
    
    try:
        # Test Segmenter API
        api_logger.info(f"[{request_id}] Testing Jina Segmenter API...")
        segments = segment_text(test_text, os.environ["JINA_API_KEY"])
        api_logger.info(
            f"[{request_id}] Segmenter test successful: "
            f"Generated {len(segments.get('chunks', []))} chunks"
        )
        
        # Test Embeddings API
        if segments.get("chunks"):
            api_logger.info(f"[{request_id}] Testing Jina Embeddings API...")
            embeddings = get_embeddings(segments["chunks"], os.environ["JINA_API_KEY"])
            api_logger.info(
                f"[{request_id}] Embeddings test successful: "
                f"Generated {len(embeddings)} embeddings"
            )
            
        return True
        
    except Exception as e:
        api_logger.error(f"[{request_id}] API test failed: {str(e)}")
        raise

def test_qdrant_connection():
    """Test Qdrant connection and basic operations"""
    api_logger = logging.getLogger('api_calls')
    request_id = hashlib.md5(f"{datetime.now()}-test-qdrant".encode()).hexdigest()[:8]
    
    try:
        # Test basic connection
        api_logger.info(f"[{request_id}] Testing Qdrant connection...")
        qdrant_url = os.environ["QDRANT_URL"]
        response = requests.get(
            f"{qdrant_url}/collections",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        response.raise_for_status()
        
        # Test collection existence
        collection_name = os.environ["QDRANT_COLLECTION_NAME"]
        response = requests.get(
            f"{qdrant_url}/collections/{collection_name}",
            headers={"api-key": os.environ["QDRANT_API_KEY"]}
        )
        
        if response.status_code == 200:
            collection_info = response.json()
            api_logger.info(
                f"[{request_id}] Collection exists: {collection_name}\n"
                f"Vector size: {collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size')}\n"
                f"Points count: {collection_info.get('points_count')}"
            )
        else:
            api_logger.warning(f"[{request_id}] Collection {collection_name} does not exist")
        
        return True
        
    except Exception as e:
        api_logger.error(f"[{request_id}] Qdrant test failed: {str(e)}")
        raise

def main():
    conn = None
    try:
        # Setup enhanced logging
        api_logger = setup_logging()
        
        # Run initial tests
        api_logger.info("Running system tests...")
        test_qdrant_connection()
        test_jina_apis()
        
        # Check environment
        check_environment()
        
        # Initialize database connection
        conn = initialize_database()
        
        # Validate Qdrant connection
        if not validate_qdrant_connection():
            raise ConnectionError("Could not connect to Qdrant server")
        
        # Get initial database stats
        initial_stats = get_database_stats(conn)
        logging.info(f"Initial database state: {initial_stats}")
        
        # Process documents
        available_docs = list_available_documents()
        if not available_docs:
            logging.warning("No documents found in docs directory")
            exit(0)
            
        selected_docs = select_documents(available_docs)
        
        # Option to force reprocess
        force_reprocess = []
        if input("Do you want to force reprocess any files? (y/n): ").lower() == 'y':
            print("\nAvailable documents:")
            for idx, doc in enumerate(selected_docs, 1):
                print(f"{idx}. {doc}")
            selection = input("\nEnter document numbers to force reprocess (comma-separated) or press Enter to skip: ")
            if selection.strip():
                try:
                    indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                    force_reprocess = [selected_docs[idx] for idx in indices if 0 <= idx < len(selected_docs)]
                except (ValueError, IndexError):
                    logging.warning("Invalid selection, proceeding without force reprocessing")
        
        # Process documents with stats
        documents, stats = process_documents(selected_docs, conn, force_reprocess)
        
        if documents:  # documents are now LangChain Document objects
            # Create Qdrant instance and upload documents
            qdrant = initialize_qdrant(documents)
            logging.info(f"Successfully uploaded {len(documents)} documents to Qdrant")
            
            # Verify random samples after upload
            num_samples = min(5, len(documents))
            test_docs = random.sample(documents, num_samples)
            api_logger = logging.getLogger('api_calls')
            request_id = hashlib.md5(f"{datetime.now()}-final-verify".encode()).hexdigest()[:8]
            
            api_logger.info(f"[{request_id}] Performing final verification with {num_samples} random samples")
            
            for i, test_doc in enumerate(test_docs, 1):
                try:
                    results = qdrant.similarity_search(test_doc.page_content, k=1)
                    if not results or results[0].page_content != test_doc.page_content:
                        error_details = {
                            "error": "Final verification failed",
                            "sample": i,
                            "expected": test_doc.page_content[:100] + "...",
                            "received": results[0].page_content[:100] + "..." if results else "No results"
                        }
                        log_qdrant_operation("Final Verification", error_details, success=False)
                        raise Exception(f"Final verification failed for sample {i}")
                    
                    api_logger.info(f"[{request_id}] Verified random sample {i}/{num_samples}")
                    
                except Exception as e:
                    api_logger.error(f"[{request_id}] Error during final verification: {str(e)}")
                    raise
            
            log_qdrant_operation("Final Verification", {
                "status": "success",
                "samples_verified": num_samples,
                "total_documents": len(documents)
            })
            logging.info("Final verification completed successfully")
            
        else:
            logging.info("No new documents to process")
        
        # Log processing summary
        stats.log_summary()
        
        # Get final database stats
        final_stats = get_database_stats(conn)
        logging.info(f"Final database state: {final_stats}")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main() 