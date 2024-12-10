"""
Data validation and quality checks for document processing.

This module provides validation functions to ensure data quality
without interrupting the existing processing flow.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
from langchain.schema import Document

# Constants for validation
MAX_DOCUMENT_SIZE_MB = 10
MIN_DOCUMENT_SIZE_BYTES = 50
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 100
SUPPORTED_FILE_TYPES = {'.md', '.txt'}
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        """Initialize validation result."""
        self.is_valid = True
        self.warnings = []
        self.errors = []
        self.stats = {}
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def update_stats(self, key: str, value: Any) -> None:
        """Update statistics."""
        self.stats[key] = value

def validate_document_file(file_path: Path) -> ValidationResult:
    """
    Validate document file before processing.
    
    Checks:
    - File exists and is readable
    - File size within limits
    - File type supported
    - Basic content validation
    """
    result = ValidationResult()
    
    try:
        # Check file exists
        if not file_path.exists():
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Check file type
        if file_path.suffix not in SUPPORTED_FILE_TYPES:
            result.add_warning(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
            )
        
        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        result.update_stats('file_size_mb', size_mb)
        
        if size_mb > MAX_DOCUMENT_SIZE_MB:
            result.add_error(
                f"File too large: {size_mb:.1f}MB "
                f"(max {MAX_DOCUMENT_SIZE_MB}MB)"
            )
        
        if file_path.stat().st_size < MIN_DOCUMENT_SIZE_BYTES:
            result.add_warning(
                f"File very small: {file_path.stat().st_size} bytes"
            )
        
        # Check content
        content = file_path.read_text()
        if not content.strip():
            result.add_error("File is empty")
        
        # Basic content stats
        result.update_stats('char_count', len(content))
        result.update_stats('line_count', len(content.splitlines()))
        result.update_stats('word_count', len(content.split()))
        
        return result
        
    except Exception as e:
        result.add_error(f"Error validating file: {str(e)}")
        return result

def validate_chunk(chunk: Document, index: int) -> ValidationResult:
    """
    Validate a single document chunk.
    
    Checks:
    - Content length within limits
    - Required metadata present
    - Content quality indicators
    """
    result = ValidationResult()
    
    try:
        # Check content length
        content_length = len(chunk.page_content)
        result.update_stats('content_length', content_length)
        
        if content_length > MAX_CHUNK_SIZE:
            result.add_warning(
                f"Chunk {index} too large: {content_length} chars"
            )
        elif content_length < MIN_CHUNK_SIZE:
            result.add_warning(
                f"Chunk {index} too small: {content_length} chars"
            )
        
        # Check required metadata
        required_fields = {'source', 'chunk_number', 'total_chunks'}
        missing_fields = required_fields - set(chunk.metadata.keys())
        if missing_fields:
            result.add_warning(
                f"Chunk {index} missing metadata: {missing_fields}"
            )
        
        # Content quality checks
        content = chunk.page_content
        if content.count('.') < 2:
            result.add_warning(
                f"Chunk {index} may be incomplete (few sentences)"
            )
        
        # Check for potential code or markup
        code_indicators = ['```', '    ', '<', '>', '{', '}']
        if any(ind in content for ind in code_indicators):
            result.add_warning(
                f"Chunk {index} may contain code or markup"
            )
        
        return result
        
    except Exception as e:
        result.add_error(f"Error validating chunk {index}: {str(e)}")
        return result

def check_duplicates(chunks: List[Document]) -> List[Tuple[int, int, float]]:
    """
    Check for potential duplicate chunks.
    Returns list of (index1, index2, similarity) for similar chunks.
    """
    duplicates = []
    
    try:
        # Create content hashes for quick comparison
        hashes = []
        for chunk in chunks:
            content = chunk.page_content.lower().strip()
            hash_obj = hashlib.md5(content.encode()).hexdigest()
            hashes.append(hash_obj)
        
        # Compare chunks
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # First check hashes
                if hashes[i] == hashes[j]:
                    duplicates.append((i, j, 1.0))
                    continue
                
                # If hashes different, do more detailed comparison
                content1 = chunks[i].page_content.lower().strip()
                content2 = chunks[j].page_content.lower().strip()
                
                # Simple similarity metric
                shorter = min(len(content1), len(content2))
                longer = max(len(content1), len(content2))
                if shorter == 0:
                    continue
                
                # Calculate similarity ratio
                similarity = shorter / longer
                if similarity > DUPLICATE_SIMILARITY_THRESHOLD:
                    duplicates.append((i, j, similarity))
        
        return duplicates
        
    except Exception as e:
        logging.error(f"Error checking duplicates: {str(e)}")
        return []

def validate_embeddings(embeddings: List[List[float]]) -> ValidationResult:
    """
    Validate embedding vectors.
    
    Checks:
    - Consistent dimensions
    - No NaN or Inf values
    - Vector magnitudes within reasonable range
    """
    result = ValidationResult()
    
    try:
        if not embeddings:
            result.add_error("No embeddings provided")
            return result
        
        # Check dimensions
        expected_dim = len(embeddings[0])
        result.update_stats('embedding_dimension', expected_dim)
        
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                result.add_error(
                    f"Inconsistent embedding dimension at index {i}: "
                    f"got {len(embedding)}, expected {expected_dim}"
                )
            
            # Check for NaN/Inf
            if not all(isinstance(x, float) and -1e6 < x < 1e6 for x in embedding):
                result.add_error(
                    f"Invalid values in embedding at index {i}"
                )
            
            # Check magnitude
            magnitude = sum(x*x for x in embedding) ** 0.5
            if not (0.1 < magnitude < 100):
                result.add_warning(
                    f"Unusual magnitude ({magnitude:.2f}) for embedding {i}"
                )
        
        return result
        
    except Exception as e:
        result.add_error(f"Error validating embeddings: {str(e)}")
        return result 