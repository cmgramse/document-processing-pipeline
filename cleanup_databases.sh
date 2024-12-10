#!/bin/bash

# Exit on error
set -e

echo "Starting database cleanup..."

# Check if environment variables are set
if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ] || [ -z "$QDRANT_COLLECTION_NAME" ]; then
    echo "Error: Required environment variables not set"
    echo "Please set: QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME"
    exit 1
fi

# Remove SQLite database files
echo "Removing SQLite database..."
rm -f data/documents.db*
echo "SQLite database removed"

# Delete Qdrant collection
echo "Deleting Qdrant collection..."
curl -X DELETE "${QDRANT_URL}/collections/${QDRANT_COLLECTION_NAME}" \
    -H "api-key: ${QDRANT_API_KEY}" \
    -H "Content-Type: application/json" || true  # Don't fail if collection doesn't exist

# Wait a moment to ensure deletion is processed
sleep 2

# Recreate Qdrant collection
echo "Recreating Qdrant collection..."
curl -X PUT "${QDRANT_URL}/collections/${QDRANT_COLLECTION_NAME}" \
    -H "api-key: ${QDRANT_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "vectors": {
            "size": 1024,
            "distance": "Cosine"
        }
    }'

echo "Database cleanup completed successfully"

# Initialize new SQLite database
echo "Initializing new SQLite database..."
python main.py init

echo "All done! Databases have been reset and reinitialized." 