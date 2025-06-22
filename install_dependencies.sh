#!/bin/bash
set -e

echo "📦 Installing Dependencies..."
echo "============================"

# Activate virtual environment
if [ ! -d "ai-comparison-env" ]; then
    echo "❌ Virtual environment not found. Run ./setup_project.sh first"
    exit 1
fi

source ai-comparison-env/bin/activate

# Install base dependencies
echo "📦 Installing base dependencies..."
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-dotenv==1.0.0
pip install pydantic==2.5.0
pip install requests==2.31.0
pip install aiofiles==23.2.1
pip install python-multipart==0.0.6

# Data handling and analysis
echo "📊 Installing data analysis packages..."
pip install pandas==2.1.4
pip install numpy==1.24.4
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install plotly==5.17.0

# Testing
echo "🧪 Installing testing packages..."
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install httpx==0.25.2

# Search-First dependencies
echo "🔍 Installing Search-First dependencies..."
pip install tavily-python==0.3.3
pip install google-generativeai==0.3.2
pip install beautifulsoup4==4.12.2

# RAG dependencies
echo "🤖 Installing RAG dependencies..."
pip install langchain==0.1.0
pip install langchain-openai==0.0.5
pip install langchain-community==0.0.12
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
pip install openai==1.6.1

# Optional: Supabase for cloud vector DB
pip install supabase==2.3.0

# Create requirements files for each system
echo "📝 Creating requirements files..."

# Search-First requirements
cat > search-first-system/requirements.txt << 'REQ_EOF'
fastapi==0.104.1
uvicorn==0.24.0
tavily-python==0.3.3
google-generativeai==0.3.2
python-dotenv==1.0.0
pydantic==2.5.0
requests==2.31.0
aiofiles==23.2.1
python-multipart==0.0.6
beautifulsoup4==4.12.2
REQ_EOF

# RAG requirements
cat > rag-system/requirements.txt << 'REQ_EOF'
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-openai==0.0.5
langchain-community==0.0.12
chromadb==0.4.22
supabase==2.3.0
sentence-transformers==2.2.2
openai==1.6.1
python-dotenv==1.0.0
pydantic==2.5.0
beautifulsoup4==4.12.2
requests==2.31.0
aiofiles==23.2.1
python-multipart==0.0.6
REQ_EOF

# Testing requirements
cat > shared-testing/requirements.txt << 'REQ_EOF'
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
pandas==2.1.4
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
REQ_EOF

echo "✅ All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Run ./create_env_files.sh to create environment templates"
echo "2. Get your API keys and update the .env files"
echo "3. Run ./create_code_templates.sh to generate code templates"
