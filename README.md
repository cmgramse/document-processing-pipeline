# Document Processing Pipeline

A robust system for processing, embedding, and storing documents using Jina AI for embeddings and Qdrant for vector storage.

## Features

- Document processing with automatic chunking
- Vector embeddings using Jina AI
- Vector storage in Qdrant
- Batch processing with optimized performance
- Comprehensive cleanup and maintenance
- Progress tracking and monitoring

## Requirements

- Python 3.8+
- SQLite3
- Jina AI API key
- Qdrant instance and API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
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
export JINA_API_KEY=your_jina_api_key
export QDRANT_API_KEY=your_qdrant_api_key
export QDRANT_URL=your_qdrant_url
export QDRANT_COLLECTION_NAME=your_collection
```

## Project Structure

```
.
├── app_documentation/     # Application documentation
├── docs/                 # Source documents to be processed
├── src/                 # Source code
│   ├── api/            # API clients (Jina, Qdrant)
│   ├── database/       # Database operations
│   ├── management/     # Document management
│   ├── monitoring/     # Metrics and monitoring
│   └── processing/     # Document processing
├── scripts/            # Utility scripts
└── tests/             # Test suite
```

## Documentation

- [Architecture Overview](app_documentation/ARCHITECTURE.md)
- [Database Schema](app_documentation/SCHEMA.md)
- [CLI Manual](app_documentation/CLI.md)

## License

[License details]
