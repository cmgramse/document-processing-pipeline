import os
import logging
from pathlib import Path

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        # Make the path absolute
        self.documents_path = Path(os.getenv('DOCUMENTS_PATH') or config.get('DOCUMENTS_PATH', 'docs')).resolve()
        
    def process_documents(self, document_paths):
        """Process the specified documents."""
        try:
            if not self.documents_path.exists():
                raise Exception(f"Documents directory not found: {self.documents_path}")
                
            for doc_path in document_paths:
                try:
                    # Convert to Path object if it's a string
                    file_path = Path(doc_path).resolve()
                    
                    if not file_path.exists():
                        logging.error(f"File not found: {file_path}")
                        continue
                        
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Process the content
                    self._process_content(content, str(file_path))
                    
                except Exception as e:
                    logging.error(f"Error reading file {doc_path}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            
    def _process_content(self, content, file_path):
        """Internal method to process document content."""
        try:
            # Add your document processing logic here
            pass
        except Exception as e:
            logging.error(f"Error processing content for {file_path}: {str(e)}") 