"""
Command Line Interface for document processing.
Provides interactive mode for document selection and processing.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from .config.settings import settings
from .processing.document_processing import DocumentProcessor
from .processing.background_tasks import BackgroundProcessor
from .database.qdrant import QdrantManager
from .api.jina import jina_client

class InteractiveProcessor:
    """Handles interactive document processing operations."""
    
    def __init__(self, conn):
        self.conn = conn
        self.doc_processor = DocumentProcessor()
        self.background_processor = BackgroundProcessor('records.db')
        self.qdrant = QdrantManager()
    
    def list_available_documents(self) -> List[str]:
        """List all available documents that can be processed."""
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
    
    def select_documents(self, available_docs: List[str]) -> List[str]:
        """Interactive document selection."""
        if not available_docs:
            raise Exception("No documents found in the docs directory")
        
        print("\nAvailable documents:")
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
    
    def force_reprocess_selection(self, selected_docs: List[str]) -> List[str]:
        """Interactive selection of documents to force reprocess."""
        print("\nDo you want to force reprocess any files? (y/n): ")
        if input().lower() != 'y':
            return []
        
        print("\nAvailable documents:")
        for idx, doc in enumerate(selected_docs, 1):
            print(f"{idx}. {doc}")
            
        selection = input("\nEnter document numbers to force reprocess (comma-separated) or press Enter to skip: ")
        if not selection.strip():
            return []
            
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            return [selected_docs[idx] for idx in indices if 0 <= idx < len(selected_docs)]
        except (ValueError, IndexError):
            logging.warning("Invalid selection, proceeding without force reprocessing")
            return []
    
    def force_reprocess_files(self, filenames: List[str]) -> None:
        """Force reprocessing by removing existing records."""
        if not filenames:
            return
            
        logging.info(f"Force reprocessing {len(filenames)} files...")
        c = self.conn.cursor()
        
        try:
            for filename in filenames:
                # Delete existing records
                c.execute('DELETE FROM documents WHERE filename = ?', (filename,))
                c.execute('DELETE FROM processed_files WHERE filename = ?', (filename,))
                logging.info(f"Cleared records for {filename}")
            
            self.conn.commit()
            logging.info(f"Successfully marked {len(filenames)} files for reprocessing")
            
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error marking files for reprocessing: {str(e)}")
            raise
    
    def process_with_progress(self, documents: List[str]) -> bool:
        """Process documents with progress bars and interactive feedback."""
        try:
            # Add documents to queue
            self.background_processor.add_documents(documents)
            
            # Start processing
            self.background_processor.start()
            
            with tqdm(total=len(documents), desc="Processing documents") as pbar:
                completed = set()
                
                while True:
                    statuses = self.background_processor.get_all_statuses()
                    active = False
                    
                    for filename, status in statuses.items():
                        if status['status'] == 'completed' and filename not in completed:
                            completed.add(filename)
                            pbar.update(1)
                        elif status['status'] != 'completed':
                            active = True
                            if status.get('error'):
                                tqdm.write(f"Error in {filename}: {status['error']}")
                    
                    if not active:
                        break
                        
                    time.sleep(1)
            
            # Final verification
            print("\nVerifying processed documents...")
            if not self.doc_processor.verify_processing(5):
                print("Warning: Verification failed for some documents")
                return False
            
            print("Processing completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            self.background_processor.stop()
            return False
            
        finally:
            self.background_processor.stop()
    
    def show_processing_summary(self) -> None:
        """Show detailed processing summary."""
        stats = self.doc_processor.get_processing_stats()
        print("\nProcessing Summary:")
        print(f"Documents: {stats.processed_documents} processed, {stats.failed_documents} failed")
        print(f"Segments: {stats.processed_segments} processed")
        print(f"Memory usage: {stats.process_memory_mb:.1f}MB")
        print(f"System memory: {stats.system_memory_percent:.1f}%")
    
    def run(self) -> bool:
        """Run the interactive processing session."""
        try:
            # List and select documents
            available_docs = self.list_available_documents()
            if not available_docs:
                print("No documents available for processing")
                return False
            
            selected_docs = self.select_documents(available_docs)
            if not selected_docs:
                print("No documents selected")
                return False
            
            # Handle force reprocessing
            force_reprocess = self.force_reprocess_selection(selected_docs)
            if force_reprocess:
                self.force_reprocess_files(force_reprocess)
            
            # Process documents
            success = self.process_with_progress(selected_docs)
            
            # Show summary
            self.show_processing_summary()
            
            return success
            
        except Exception as e:
            logging.error(f"Error in interactive processing: {str(e)}")
            return False

def main(conn) -> bool:
    """Main CLI entry point."""
    try:
        processor = InteractiveProcessor(conn)
        return processor.run()
        
    except Exception as e:
        logging.error(f"CLI error: {str(e)}")
        return False 