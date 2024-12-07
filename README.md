# Document Processing Pipeline

A robust Python-based document processing pipeline that processes markdown documents, segments them into chunks, generates embeddings using Jina AI, and stores them in Qdrant vector database.

## Features

- Document segmentation using Jina AI
- Vector embeddings generation
- Qdrant vector database integration
- SQLite-based progress tracking
- Robust error handling and retry mechanisms
- Comprehensive document management
- Batch processing optimization
- Efficient cleanup strategies

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Process Documents
```bash
# Process new documents
python main.py process

# Force reprocess specific documents
python main.py process --force-reprocess doc1.md doc2.md
```

### Manage Documents
```bash
# Delete documents
python main.py delete doc1.md doc2.md

# View document statistics
python main.py stats

# Clean up old processed chunks
python main.py cleanup --retention-days 30
```

### Document Selection Options
- Individual numbers: "1,3,5"
- Ranges: "1-5"
- Combinations: "1,3-5,7"
- All documents: "all"
- Latest N documents: "latest:N"

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_document_manager.py
```

## Project Structure

```
.
├── docs/                  # Document storage
├── src/
│   ├── api/              # API integrations
│   ├── database/         # Database operations
│   ├── management/       # Document management
│   ├── processing/       # Document processing
│   └── testing/          # API tests
├── tests/                # Test suite
├── main.py              # Main application
└── requirements.txt     # Dependencies
```

## Environment Variables

Required environment variables:
- `JINA_API_KEY`: Your Jina AI API key
- `QDRANT_API_KEY`: Your Qdrant API key
- `QDRANT_URL`: Qdrant server URL
- `QDRANT_COLLECTION_NAME`: Name of your Qdrant collection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
