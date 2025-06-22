#!/bin/bash

echo "🤖 Starting RAG System..."
echo "========================"

# Check if we're in the right directory
if [ ! -d "rag-system" ]; then
    echo "❌ rag-system directory not found"
    echo "Run this script from the ai-comparison-project root directory"
    exit 1
fi

# Activate virtual environment
if [ ! -d "ai-comparison-env" ]; then
    echo "❌ Virtual environment not found. Run ./setup_project.sh first"
    exit 1
fi

source ai-comparison-env/bin/activate

# Check if .env file exists
if [ ! -f "rag-system/.env" ]; then
    echo "❌ .env file not found in rag-system/"
    echo "Run ./create_env_files.sh and add your API keys"
    exit 1
fi

# Check API keys
cd rag-system

# Source the .env file to check variables
set -a
source .env
set +a

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ OPENAI_API_KEY not configured in .env file"
    echo "Get your API key from https://platform.openai.com"
    exit 1
fi

echo "✅ Environment configured"
echo "✅ API keys found"

# Create data directories if they don't exist
mkdir -p data/{raw_docs,processed,chromadb}

# Start the server
echo ""
echo "🚀 Starting RAG server on http://localhost:8002"
echo "Press Ctrl+C to stop"
echo ""

cd src
python main.py
