#!/bin/bash

echo "🔧 Creating Environment Files..."
echo "==============================="

# Search-First .env.example
cat > search-first-system/.env.example << 'ENV_EOF'
# Search-First System Environment Variables
# ========================================

# Required API Keys
TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Search Configuration
MAX_SEARCH_RESULTS=10
SEARCH_DEPTH=basic
MAX_CONTENT_SIZE=100000
ENABLE_SEARCH_SUMMARIZATION=false

# Allowed Domains for Search (comma-separated)
ALLOWED_DOMAINS=docs.python.org,fastapi.tiangolo.com,python.langchain.com,react.dev

# LLM Configuration
LLM_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=10000

# Server Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=true
LOG_LEVEL=INFO

# Timeouts
REQUEST_TIMEOUT=30
LLM_TIMEOUT=60
SEARCH_TIMEOUT=15

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_CONCURRENT_REQUESTS=10
ENV_EOF

# RAG .env.example
cat > rag-system/.env.example << 'ENV_EOF'
# RAG System Environment Variables
# ===============================

# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMADB_PATH=./data/chromadb
COLLECTION_NAME=documentation_chunks

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENTS=1000
SUPPORTED_FILE_TYPES=.txt,.md,.pdf,.html

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Retrieval Configuration
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.7
RETRIEVAL_METHOD=similarity

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=10000

# Server Configuration
HOST=0.0.0.0
PORT=8002
DEBUG=true
LOG_LEVEL=INFO

# Document Sources URLs (comma-separated)
DOCUMENT_URLS=https://docs.python.org/3/,https://fastapi.tiangolo.com/,https://python.langchain.com/

# Processing Configuration
BATCH_SIZE=50
MAX_WORKERS=4
ENV_EOF

# Copy .env.example to .env for both systems
cp search-first-system/.env.example search-first-system/.env
cp rag-system/.env.example rag-system/.env

echo "✅ Environment files created!"
echo ""
echo "📝 Don't forget to:"
echo "1. Get API keys from:"
echo "   - Tavily: https://tavily.com"
echo "   - Google AI: https://ai.google.dev"
echo "   - OpenAI: https://platform.openai.com"
echo "   - Supabase (optional): https://supabase.com"
echo ""
echo "2. Update the .env files with your actual API keys"
echo ""
echo "Next: Run ./create_code_templates.sh"
