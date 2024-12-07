import time
from typing import Dict, Any
import logging

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