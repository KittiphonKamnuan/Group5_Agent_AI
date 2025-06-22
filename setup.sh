#!/bin/bash
# ==============================================================================
# AI COMPARISON PROJECT - COMPLETE SETUP SCRIPTS
# ==============================================================================

# ==============================================================================
# 1. setup_project.sh - Main project setup script
# ==============================================================================
cat > setup_project.sh << 'EOF'
#!/bin/bash
set -e  # Exit on any error

echo "🚀 Setting up AI Comparison Project..."
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is too old. Please install Python 3.9+"
    exit 1
fi

# Create project structure
echo "📁 Creating project structure..."
mkdir -p ai-comparison-project
cd ai-comparison-project

# Create main directories
mkdir -p {search-first-system,rag-system,shared-testing,evaluation-results,docs}

# Create subdirectories
mkdir -p search-first-system/{src,tests,config}
mkdir -p rag-system/{src,tests,config,data}
mkdir -p shared-testing/{test_data,metrics,reports}
mkdir -p evaluation-results/{charts,raw_data,analysis}

# Create data directories for RAG
mkdir -p rag-system/data/{raw_docs,processed,chromadb}

echo "✅ Project structure created successfully!"

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv ai-comparison-env

# Activate virtual environment
source ai-comparison-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "✅ Virtual environment created and activated!"

# Create .gitignore
cat > .gitignore << 'GITIGNORE_EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
ai-comparison-env/
venv/
env/

# Environment Variables
.env
*.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
*.csv
*.json
!test_questions.json
!package.json

# Logs
*.log
logs/

# Database
*.db
*.sqlite3
chromadb/

# Results
evaluation-results/raw_data/*.csv
evaluation-results/charts/*.png
evaluation-results/analysis/*.html

# OS
.DS_Store
Thumbs.db
GITIGNORE_EOF

echo "✅ .gitignore created!"
echo ""
echo "🎉 Basic setup complete!"
echo "Next: Run ./install_dependencies.sh"
EOF

chmod +x setup_project.sh

# ==============================================================================
# 2. install_dependencies.sh - Install all required packages
# ==============================================================================
cat > install_dependencies.sh << 'EOF'
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
EOF

chmod +x install_dependencies.sh

# ==============================================================================
# 3. create_env_files.sh - Create environment configuration files
# ==============================================================================
cat > create_env_files.sh << 'EOF'
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
EOF

chmod +x create_env_files.sh

# ==============================================================================
# 4. create_code_templates.sh - Generate code templates
# ==============================================================================
cat > create_code_templates.sh << 'EOF'
#!/bin/bash

echo "💻 Creating Code Templates..."
echo "============================"

# Create Search-First main application
cat > search-first-system/src/main.py << 'CODE_EOF'
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Search-First Q&A System",
    description="AI Q&A system using search-first approach",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 10
    include_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time: float
    method: str = "search-first"
    metadata: Dict
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    system: str
    timestamp: float
    version: str

# Global variables for tracking
request_count = 0
total_response_time = 0.0

@app.on_startup
async def startup_event():
    """Initialize the application"""
    print("🚀 Search-First System Starting...")
    
    # Verify API keys
    tavily_key = os.getenv("TAVILY_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        print("❌ TAVILY_API_KEY not configured")
    else:
        print("✅ Tavily API key configured")
    
    if not google_key or google_key == "your_google_api_key_here":
        print("❌ GOOGLE_API_KEY not configured")
    else:
        print("✅ Google API key configured")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global request_count, total_response_time
    
    avg_response_time = total_response_time / max(request_count, 1)
    
    return HealthResponse(
        status="healthy",
        system="search-first",
        timestamp=time.time(),
        version="1.0.0"
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    global request_count, total_response_time
    
    return {
        "total_requests": request_count,
        "average_response_time": total_response_time / max(request_count, 1),
        "uptime": time.time()
    }

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint"""
    global request_count, total_response_time
    
    start_time = time.time()
    request_count += 1
    
    try:
        # TODO: Implement actual search-first logic
        # For now, return a placeholder response
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        response_time = time.time() - start_time
        total_response_time += response_time
        
        result = QueryResponse(
            answer="This is a placeholder response. Implement search-first logic in the TODO section.",
            sources=["https://example.com/placeholder"],
            response_time=response_time,
            metadata={
                "query_length": len(request.question),
                "max_results": request.max_results,
                "timestamp": start_time
            },
            confidence=0.8
        )
        
        return result
    
    except Exception as e:
        response_time = time.time() - start_time
        total_response_time += response_time
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/reset")
async def reset_conversation():
    """Reset conversation state"""
    return {"message": "Conversation reset (if stateful implementation exists)"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
CODE_EOF

# Create RAG main application
cat > rag-system/src/main.py << 'CODE_EOF'
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import json
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RAG Q&A System",
    description="AI Q&A system using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time: float
    method: str = "rag"
    metadata: Dict
    retrieved_chunks: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    system: str
    timestamp: float
    version: str
    vector_db_status: str

class DocumentRequest(BaseModel):
    urls: List[str]
    force_reprocess: Optional[bool] = False

# Global variables
request_count = 0
total_response_time = 0.0
vector_db_initialized = False

@app.on_startup
async def startup_event():
    """Initialize the RAG system"""
    global vector_db_initialized
    
    print("🚀 RAG System Starting...")
    
    # Verify API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not configured")
    else:
        print("✅ OpenAI API key configured")
    
    # TODO: Initialize vector database
    # vector_db_initialized = True
    print("⚠️  Vector database initialization needed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global vector_db_initialized
    
    return HealthResponse(
        status="healthy",
        system="rag",
        timestamp=time.time(),
        version="1.0.0",
        vector_db_status="initialized" if vector_db_initialized else "not_initialized"
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    global request_count, total_response_time
    
    return {
        "total_requests": request_count,
        "average_response_time": total_response_time / max(request_count, 1),
        "vector_db_initialized": vector_db_initialized,
        "uptime": time.time()
    }

@app.post("/process_documents")
async def process_documents(request: DocumentRequest):
    """Process and index documents"""
    try:
        # TODO: Implement document processing pipeline
        # 1. Download/load documents
        # 2. Chunk documents
        # 3. Generate embeddings
        # 4. Store in vector database
        
        return {
            "message": "Document processing not implemented yet",
            "urls_requested": request.urls,
            "status": "pending_implementation"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint using RAG"""
    global request_count, total_response_time
    
    start_time = time.time()
    request_count += 1
    
    try:
        # TODO: Implement RAG pipeline
        # 1. Generate query embedding
        # 2. Search vector database
        # 3. Retrieve relevant chunks
        # 4. Generate response with LLM
        
        # Simulate processing time
        await asyncio.sleep(1.0)  # RAG typically slower than search
        
        response_time = time.time() - start_time
        total_response_time += response_time
        
        result = QueryResponse(
            answer="This is a placeholder RAG response. Implement the RAG pipeline in the TODO section.",
            sources=["Document chunk 1", "Document chunk 2"],
            response_time=response_time,
            metadata={
                "query_length": len(request.question),
                "retrieval_k": request.k,
                "similarity_threshold": request.similarity_threshold,
                "timestamp": start_time
            },
            retrieved_chunks=request.k
        )
        
        return result
    
    except Exception as e:
        response_time = time.time() - start_time
        total_response_time += response_time
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

@app.post("/reset")
async def reset_conversation():
    """Reset conversation state"""
    return {"message": "Conversation reset (if stateful implementation exists)"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8002))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
CODE_EOF

# Create test questions
cat > shared-testing/test_questions.json << 'JSON_EOF'
{
  "basic_questions": [
    {
      "id": "basic_001",
      "question": "How to create a FastAPI route?",
      "category": "basic",
      "expected_topics": ["fastapi", "routing", "decorator"],
      "difficulty": 1
    },
    {
      "id": "basic_002",
      "question": "What is React useState hook?",
      "category": "basic", 
      "expected_topics": ["react", "hooks", "state"],
      "difficulty": 1
    },
    {
      "id": "basic_003",
      "question": "How to import modules in Python?",
      "category": "basic",
      "expected_topics": ["python", "import", "modules"],
      "difficulty": 1
    },
    {
      "id": "basic_004",
      "question": "What is LangChain?",
      "category": "basic",
      "expected_topics": ["langchain", "framework", "llm"],
      "difficulty": 1
    },
    {
      "id": "basic_005",
      "question": "How to handle CORS in FastAPI?",
      "category": "basic",
      "expected_topics": ["fastapi", "cors", "middleware"],
      "difficulty": 1
    }
  ],
  "intermediate_questions": [
    {
      "id": "int_001",
      "question": "How to implement authentication in FastAPI with JWT tokens?",
      "category": "intermediate",
      "expected_topics": ["fastapi", "authentication", "jwt", "security"],
      "difficulty": 3
    },
    {
      "id": "int_002",
      "question": "Best practices for React component optimization and performance?",
      "category": "intermediate",
      "expected_topics": ["react", "performance", "optimization", "memo"],
      "difficulty": 3
    },
    {
      "id": "int_003",
      "question": "How to create custom LangChain agents with tools?",
      "category": "intermediate",
      "expected_topics": ["langchain", "agents", "tools", "custom"],
      "difficulty": 3
    },
    {
      "id": "int_004",
      "question": "How to handle database operations in FastAPI with SQLAlchemy?",
      "category": "intermediate",
      "expected_topics": ["fastapi", "database", "sqlalchemy", "orm"],
      "difficulty": 3
    },
    {
      "id": "int_005",
      "question": "How to implement caching in Python applications?",
      "category": "intermediate",
      "expected_topics": ["python", "caching", "redis", "performance"],
      "difficulty": 3
    }
  ],
  "complex_questions": [
    {
      "id": "comp_001",
      "question": "Build a production-ready FastAPI application with database, authentication, testing, and deployment",
      "category": "complex",
      "expected_topics": ["fastapi", "database", "authentication", "testing", "production", "deployment"],
      "difficulty": 5
    },
    {
      "id": "comp_002",
      "question": "Create a scalable React application with advanced state management, routing, and performance optimization",
      "category": "complex",
      "expected_topics": ["react", "scalability", "state-management", "routing", "performance"],
      "difficulty": 5
    },
    {
      "id": "comp_003",
      "question": "Design and implement a multi-agent system using LangChain with memory, tools, and coordination",
      "category": "complex",
      "expected_topics": ["langchain", "multi-agent", "memory", "tools", "coordination"],
      "difficulty": 5
    },
    {
      "id": "comp_004",
      "question": "How to implement a complete RAG system with document processing, embeddings, and vector search?",
      "category": "complex",
      "expected_topics": ["rag", "embeddings", "vector-search", "document-processing"],
      "difficulty": 5
    },
    {
      "id": "comp_005",
      "question": "Build a microservices architecture with FastAPI, Docker, and Kubernetes",
      "category": "complex",
      "expected_topics": ["microservices", "fastapi", "docker", "kubernetes", "architecture"],
      "difficulty": 5
    }
  ]
}
JSON_EOF

# Create evaluator script
cat > shared-testing/evaluator.py << 'EVAL_EOF'
import asyncio
import httpx
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
from datetime import datetime

class SystemEvaluator:
    def __init__(self):
        self.results = []
        self.search_first_url = "http://localhost:8001"
        self.rag_url = "http://localhost:8002"
        
    async def check_system_health(self, name: str, url: str):
        """Check if system is running and healthy"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=10.0)
                if response.status_code == 200:
                    print(f"✅ {name} system is healthy")
                    return True
                else:
                    print(f"❌ {name} system returned status {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ {name} system is not responding: {str(e)}")
            return False
    
    async def test_system(self, system_name: str, base_url: str, questions: List[Dict]):
        """Test a system with given questions"""
        print(f"\n🧪 Testing {system_name} system...")
        
        async with httpx.AsyncClient() as client:
            for i, question_data in enumerate(questions):
                print(f"  Question {i+1}/{len(questions)}: {question_data['id']}")
                
                start_time = time.time()
                
                try:
                    response = await client.post(
                        f"{base_url}/chat",
                        json={"question": question_data["question"]},
                        timeout=120.0
                    )
                    
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        evaluation_result = {
                            "system": system_name,
                            "question_id": question_data["id"],
                            "question": question_data["question"],
                            "category": question_data["category"],
                            "difficulty": question_data.get("difficulty", 1),
                            "answer": result["answer"],
                            "sources": result.get("sources", []),
                            "response_time": end_time - start_time,
                            "api_response_time": result.get("response_time", 0),
                            "success": True,
                            "error": None,
                            "timestamp": time.time(),
                            "metadata": result.get("metadata", {})
                        }
                        print(f"    ✅ Success ({end_time - start_time:.2f}s)")
                    else:
                        evaluation_result = {
                            "system": system_name,
                            "question_id": question_data["id"],
                            "question": question_data["question"],
                            "category": question_data["category"],
                            "difficulty": question_data.get("difficulty", 1),
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": end_time - start_time,
                            "timestamp": time.time()
                        }
                        print(f"    ❌ Failed: HTTP {response.status_code}")
                
                except Exception as e:
                    evaluation_result = {
                        "system": system_name,
                        "question_id": question_data["id"],
                        "question": question_data["question"],
                        "category": question_data["category"],
                        "difficulty": question_data.get("difficulty", 1),
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time,
                        "timestamp": time.time()
                    }
                    print(f"    ❌ Error: {str(e)}")
                
                self.results.append(evaluation_result)
                
                # Small delay between requests
                await asyncio.sleep(2)
    
    async def run_comparison(self, questions_file: str = "test_questions.json"):
        """Run comparison between both systems"""
        print("🚀 Starting AI Systems Comparison")
        print("=" * 50)
        
        # Check if both systems are running
        search_healthy = await self.check_system_health("Search-First", self.search_first_url)
        rag_healthy = await self.check_system_health("RAG", self.rag_url)
        
        if not search_healthy or not rag_healthy:
            print("\n❌ One or both systems are not running. Please start them first:")
            print("   Search-First: ./search-first-system/start.sh")
            print("   RAG: ./rag-system/start.sh")
            return None
        
        # Load questions
        try:
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Questions file {questions_file} not found")
            return None
        
        # Flatten questions
        all_questions = []
        for category, questions in questions_data.items():
            all_questions.extend(questions)
        
        print(f"\n📋 Running evaluation with {len(all_questions)} questions...")
        
        # Test Search-First
        await self.test_system("search_first", self.search_first_url, all_questions)
        
        # Test RAG  
        await self.test_system("rag", self.rag_url, all_questions)
        
        # Save results
        df = pd.DataFrame(self.results)
        
        # Create output directory
        os.makedirs("../evaluation-results/raw_data", exist_ok=True)
        os.makedirs("../evaluation-results/charts", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"../evaluation-results/raw_data/comparison_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        print(f"\n✅ Evaluation complete! Results saved to {results_file}")
        return df
    
    def analyze_results(self, df: pd.DataFrame = None):
        """Analyze and visualize results"""
        if df is None:
            df = pd.DataFrame(self.results)
        
        if df.empty:
            print("❌ No results to analyze")
            return
        
        print("\n📊 RESULTS ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        success_rate = df.groupby('system')['success'].mean() * 100
        avg_response_time = df[df['success']].groupby('system')['response_time'].mean()
        
        print("\n📈 Overall Performance:")
        for system in df['system'].unique():
            successful_queries = df[(df['system'] == system) & (df['success'] == True)]
            failed_queries = df[(df['system'] == system) & (df['success'] == False)]
            
            print(f"\n{system.upper().replace('_', '-')}:")
            print(f"  ✅ Success Rate: {success_rate[system]:.1f}%")
            print(f"  ⏱️  Avg Response Time: {avg_response_time[system]:.2f}s")
            print(f"  📊 Total Queries: {len(df[df['system'] == system])}")
            print(f"  ✅ Successful: {len(successful_queries)}")
            print(f"  ❌ Failed: {len(failed_queries)}")
        
        # Category breakdown
        print(f"\n📋 Performance by Category:")
        category_stats = df[df['success']].groupby(['system', 'category']).agg({
            'response_time': 'mean',
            'success': 'count'
        }).round(2)
        print(category_stats)
        
        # Create visualizations
        self.create_visualizations(df)
        
        return df
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create comparison charts"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Systems Comparison: RAG vs Search-First', fontsize=16, fontweight='bold')
        
        # 1. Response Time Comparison
        successful_df = df[df['success']]
        sns.boxplot(data=successful_df, x='system', y='response_time', ax=axes[0,0])
        axes[0,0].set_title('Response Time Distribution')
        axes[0,0].set_ylabel('Response Time (seconds)')
        axes[0,0].set_xlabel('System')
        
        # 2. Success Rate by Category
        success_by_category = df.groupby(['system', 'category'])['success'].mean().reset_index()
        sns.barplot(data=success_by_category, x='category', y='success', hue='system', ax=axes[0,1])
        axes[0,1].set_title('Success Rate by Question Category')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Response Time by Category
        sns.barplot(data=successful_df, x='category', y='response_time', hue='system', ax=axes[1,0])
        axes[1,0].set_title('Average Response Time by Category')
        axes[1,0].set_ylabel('Response Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Success Rate Comparison
        overall_success = df.groupby('system')['success'].mean()
        axes[1,1].bar(overall_success.index, overall_success.values)
        axes[1,1].set_title('Overall Success Rate')
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = f'../evaluation-results/charts/comparison_analysis_{timestamp}.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Charts saved to {chart_file}")
        plt.show()

async def main():
    """Main evaluation function"""
    evaluator = SystemEvaluator()
    results_df = await evaluator.run_comparison()
    
    if results_df is not None:
        evaluator.analyze_results(results_df)
    else:
        print("❌ Evaluation failed - check that both systems are running")

if __name__ == "__main__":
    asyncio.run(main())
EVAL_EOF

echo "✅ Code templates created successfully!"
echo ""
echo "📁 Created files:"
echo "  - search-first-system/src/main.py"
echo "  - rag-system/src/main.py"
echo "  - shared-testing/test_questions.json"
echo "  - shared-testing/evaluator.py"
echo ""
echo "Next: Set up your API keys and run the systems!"
EOF

chmod +x create_code_templates.sh

# ==============================================================================
# 5. start_search_first.sh - Start Search-First system
# ==============================================================================
cat > start_search_first.sh << 'EOF'
#!/bin/bash

echo "🔍 Starting Search-First System..."
echo "================================="

# Check if we're in the right directory
if [ ! -d "search-first-system" ]; then
    echo "❌ search-first-system directory not found"
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
if [ ! -f "search-first-system/.env" ]; then
    echo "❌ .env file not found in search-first-system/"
    echo "Run ./create_env_files.sh and add your API keys"
    exit 1
fi

# Check API keys
cd search-first-system

# Source the .env file to check variables
set -a
source .env
set +a

if [ -z "$TAVILY_API_KEY" ] || [ "$TAVILY_API_KEY" = "your_tavily_api_key_here" ]; then
    echo "❌ TAVILY_API_KEY not configured in .env file"
    echo "Get your API key from https://tavily.com"
    exit 1
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "❌ GOOGLE_API_KEY not configured in .env file"
    echo "Get your API key from https://ai.google.dev"
    exit 1
fi

echo "✅ Environment configured"
echo "✅ API keys found"

# Start the server
echo ""
echo "🚀 Starting Search-First server on http://localhost:8001"
echo "Press Ctrl+C to stop"
echo ""

cd src
python main.py
EOF

chmod +x start_search_first.sh

# ==============================================================================
# 6. start_rag.sh - Start RAG system  
# ==============================================================================
cat > start_rag.sh << 'EOF'
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
EOF

chmod +x start_rag.sh

# ==============================================================================
# 7. run_comparison.sh - Run the comparison test
# ==============================================================================
cat > run_comparison.sh << 'EOF'
#!/bin/bash

echo "🔄 Running AI Systems Comparison..."
echo "=================================="

# Activate virtual environment
if [ ! -d "ai-comparison-env" ]; then
    echo "❌ Virtual environment not found. Run ./setup_project.sh first"
    exit 1
fi

source ai-comparison-env/bin/activate

# Check if both systems are running
echo "🔍 Checking if systems are running..."

# Check Search-First
search_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health || echo "000")
if [ "$search_response" != "200" ]; then
    echo "❌ Search-First system not running on port 8001"
    echo "Start it with: ./start_search_first.sh"
    exit 1
fi
echo "✅ Search-First system is running"

# Check RAG
rag_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/health || echo "000")
if [ "$rag_response" != "200" ]; then
    echo "❌ RAG system not running on port 8002"
    echo "Start it with: ./start_rag.sh"
    exit 1
fi
echo "✅ RAG system is running"

# Run the comparison
echo ""
echo "🚀 Starting comparison test..."
echo "This may take several minutes..."
echo ""

cd shared-testing
python evaluator.py

echo ""
echo "✅ Comparison complete!"
echo "📊 Check evaluation-results/ folder for detailed results"
EOF

chmod +x run_comparison.sh

# ==============================================================================
# 8. test_systems.sh - Quick test both systems
# ==============================================================================
cat > test_systems.sh << 'EOF'
#!/bin/bash

echo "🧪 Quick System Test..."
echo "====================="

# Function to test a system
test_system() {
    local name=$1
    local url=$2
    local port=$3
    
    echo "Testing $name system..."
    
    # Health check
    health_response=$(curl -s -w "%{http_code}" "$url/health" || echo "ERROR")
    if [[ $health_response == *"200" ]]; then
        echo "  ✅ Health check passed"
    else
        echo "  ❌ Health check failed"
        return 1
    fi
    
    # Quick chat test
    echo "  🔄 Testing chat endpoint..."
    chat_response=$(curl -s -X POST "$url/chat" \
        -H "Content-Type: application/json" \
        -d '{"question":"What is Python?"}' \
        -w "%{http_code}" || echo "ERROR")
    
    if [[ $chat_response == *"200" ]]; then
        echo "  ✅ Chat endpoint working"
    else
        echo "  ❌ Chat endpoint failed"
        return 1
    fi
    
    echo "  ✅ $name system is working correctly"
    return 0
}

# Test both systems
echo "🔍 Testing Search-First system (port 8001)..."
test_system "Search-First" "http://localhost:8001" 8001

echo ""
echo "🤖 Testing RAG system (port 8002)..."
test_system "RAG" "http://localhost:8002" 8002

echo ""
echo "🎉 System testing complete!"
echo ""
echo "If tests passed, you can run the full comparison with:"
echo "./run_comparison.sh"
EOF

chmod +x test_systems.sh

# ==============================================================================
# 9. stop_systems.sh - Stop all running systems
# ==============================================================================
cat > stop_systems.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping AI Systems..."
echo "========================"

# Find and kill processes on ports 8001 and 8002
echo "🔍 Looking for running systems..."

# Kill Search-First (port 8001)
search_pid=$(lsof -ti:8001 2>/dev/null)
if [ -n "$search_pid" ]; then
    echo "🛑 Stopping Search-First system (PID: $search_pid)"
    kill -TERM $search_pid 2>/dev/null || kill -KILL $search_pid 2>/dev/null
    sleep 2
    if kill -0 $search_pid 2>/dev/null; then
        echo "  ⚠️  Force killing Search-First system"
        kill -KILL $search_pid 2>/dev/null
    fi
    echo "  ✅ Search-First system stopped"
else
    echo "  ℹ️  Search-First system not running"
fi

# Kill RAG (port 8002)
rag_pid=$(lsof -ti:8002 2>/dev/null)
if [ -n "$rag_pid" ]; then
    echo "🛑 Stopping RAG system (PID: $rag_pid)"
    kill -TERM $rag_pid 2>/dev/null || kill -KILL $rag_pid 2>/dev/null
    sleep 2
    if kill -0 $rag_pid 2>/dev/null; then
        echo "  ⚠️  Force killing RAG system"
        kill -KILL $rag_pid 2>/dev/null
    fi
    echo "  ✅ RAG system stopped"
else
    echo "  ℹ️  RAG system not running"
fi

# Also kill any Python processes that might be hanging
echo "🧹 Cleaning up any remaining Python processes..."
pkill -f "main.py" 2>/dev/null || true

echo ""
echo "✅ All systems stopped"
EOF

chmod +x stop_systems.sh

# ==============================================================================
# 10. verify_setup.sh - Verify the complete setup
# ==============================================================================
cat > verify_setup.sh << 'EOF'
#!/bin/bash

echo "🔍 Verifying Complete Setup..."
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success_count=0
total_checks=0

check_item() {
    local item=$1
    local condition=$2
    total_checks=$((total_checks + 1))
    
    if eval "$condition"; then
        echo -e "  ${GREEN}✅${NC} $item"
        success_count=$((success_count + 1))
    else
        echo -e "  ${RED}❌${NC} $item"
    fi
}

echo "🐍 Python Environment:"
check_item "Python 3.9+ installed" "python3 --version | grep -E 'Python 3\.(9|[1-9][0-9])'"
check_item "Virtual environment exists" "[ -d 'ai-comparison-env' ]"
check_item "Virtual environment activatable" "source ai-comparison-env/bin/activate && python --version"

echo ""
echo "📁 Project Structure:"
check_item "Search-First directory" "[ -d 'search-first-system' ]"
check_item "RAG directory" "[ -d 'rag-system' ]"
check_item "Testing directory" "[ -d 'shared-testing' ]"
check_item "Results directory" "[ -d 'evaluation-results' ]"

echo ""
echo "📄 Configuration Files:"
check_item "Search-First .env" "[ -f 'search-first-system/.env' ]"
check_item "RAG .env" "[ -f 'rag-system/.env' ]"
check_item "Test questions" "[ -f 'shared-testing/test_questions.json' ]"

echo ""
echo "💻 Code Files:"
check_item "Search-First main.py" "[ -f 'search-first-system/src/main.py' ]"
check_item "RAG main.py" "[ -f 'rag-system/src/main.py' ]"
check_item "Evaluator script" "[ -f 'shared-testing/evaluator.py' ]"

echo ""
echo "🔧 Executable Scripts:"
check_item "Start Search-First script" "[ -x './start_search_first.sh' ]"
check_item "Start RAG script" "[ -x './start_rag.sh' ]"
check_item "Run comparison script" "[ -x './run_comparison.sh' ]"
check_item "Test systems script" "[ -x './test_systems.sh' ]"
check_item "Stop systems script" "[ -x './stop_systems.sh' ]"

echo ""
echo "🔑 API Keys Check:"
if [ -f "search-first-system/.env" ]; then
    source search-first-system/.env
    check_item "Tavily API key configured" "[ '$TAVILY_API_KEY' != 'your_tavily_api_key_here' ] && [ -n '$TAVILY_API_KEY' ]"
    check_item "Google API key configured" "[ '$GOOGLE_API_KEY' != 'your_google_api_key_here' ] && [ -n '$GOOGLE_API_KEY' ]"
fi

if [ -f "rag-system/.env" ]; then
    source rag-system/.env
    check_item "OpenAI API key configured" "[ '$OPENAI_API_KEY' != 'your_openai_api_key_here' ] && [ -n '$OPENAI_API_KEY' ]"
fi

echo ""
echo "📦 Dependencies Check:"
source ai-comparison-env/bin/activate 2>/dev/null
check_item "FastAPI installed" "python -c 'import fastapi' 2>/dev/null"
check_item "Tavily installed" "python -c 'import tavily' 2>/dev/null"
check_item "Google GenAI installed" "python -c 'import google.generativeai' 2>/dev/null"
check_item "LangChain installed" "python -c 'import langchain' 2>/dev/null"
check_item "Pandas installed" "python -c 'import pandas' 2>/dev/null"

echo ""
echo "📊 Summary:"
echo "==========="
percentage=$((success_count * 100 / total_checks))

if [ $percentage -eq 100 ]; then
    echo -e "${GREEN}🎉 Perfect! All $total_checks checks passed (100%)${NC}"
    echo ""
    echo "🚀 You're ready to start comparing systems!"
    echo ""
    echo "Quick start commands:"
    echo "  ./start_search_first.sh    # Terminal 1"
    echo "  ./start_rag.sh             # Terminal 2" 
    echo "  ./test_systems.sh          # Test both"
    echo "  ./run_comparison.sh        # Full comparison"
elif [ $percentage -ge 80 ]; then
    echo -e "${YELLOW}⚠️  Good setup: $success_count/$total_checks checks passed ($percentage%)${NC}"
    echo "You can proceed, but consider fixing the failed items."
else
    echo -e "${RED}❌ Setup needs work: $success_count/$total_checks checks passed ($percentage%)${NC}"
    echo "Please fix the failed items before proceeding."
fi

echo ""
echo "📚 Need help? Check the docs/ folder or run:"
echo "  ./setup_project.sh          # Redo basic setup"
echo "  ./create_env_files.sh       # Recreate config files"
echo "  ./install_dependencies.sh   # Reinstall packages"
EOF

chmod +x verify_setup.sh

# ==============================================================================
# 11. clean_project.sh - Clean up project (optional)
# ==============================================================================
cat > clean_project.sh << 'EOF'
#!/bin/bash

echo "🧹 Cleaning AI Comparison Project..."
echo "===================================="

# Ask for confirmation
read -p "This will remove all generated files and results. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Stop any running systems
echo "🛑 Stopping running systems..."
./stop_systems.sh 2>/dev/null || true

# Clean up results
echo "🗑️  Removing evaluation results..."
rm -rf evaluation-results/raw_data/*.csv
rm -rf evaluation-results/charts/*.png
rm -rf evaluation-results/analysis/*.html

# Clean up generated data
echo "🗑️  Removing generated data..."
rm -rf rag-system/data/chromadb/*
rm -rf rag-system/data/processed/*

# Clean up Python cache
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Clean up logs
echo "🗑️  Removing logs..."
find . -name "*.log" -delete 2>/dev/null || true

echo ""
echo "✅ Project cleaned!"
echo ""
echo "The following were preserved:"
echo "  - Source code"
echo "  - Configuration files (.env)"
echo "  - Virtual environment"
echo "  - Test questions"
echo ""
echo "To start fresh, run: ./run_comparison.sh"
EOF

chmod +x clean_project.sh

# ==============================================================================
# 12. complete_setup.sh - Run all setup steps
# ==============================================================================
cat > complete_setup.sh << 'EOF'
#!/bin/bash

echo "🚀 Complete AI Comparison Project Setup"
echo "========================================"
echo ""
echo "This script will set up everything you need to compare RAG vs Search-First systems."
echo ""

# Ask for confirmation
read -p "Continue with complete setup? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "❌ Setup cancelled"
    exit 0
fi

# Run setup steps
echo "📁 Step 1: Creating project structure..."
./setup_project.sh

echo ""
echo "📦 Step 2: Installing dependencies..."
./install_dependencies.sh

echo ""
echo "🔧 Step 3: Creating environment files..."
./create_env_files.sh

echo ""
echo "💻 Step 4: Creating code templates..."
./create_code_templates.sh

echo ""
echo "🔍 Step 5: Verifying setup..."
./verify_setup.sh

echo ""
echo "🎉 Complete setup finished!"
echo ""
echo "📝 Next steps:"
echo "1. Get your API keys:"
echo "   - Tavily: https://tavily.com"
echo "   - Google AI: https://ai.google.dev"
echo "   - OpenAI: https://platform.openai.com"
echo ""
echo "2. Update your .env files with the API keys"
echo ""
echo "3. Start the systems and run comparison:"
echo "   ./start_search_first.sh    # Terminal 1"
echo "   ./start_rag.sh             # Terminal 2"
echo "   ./run_comparison.sh        # Terminal 3"
echo ""
echo "Happy comparing! 🚀"
EOF

chmod +x complete_setup.sh

echo "✅ All shell scripts created successfully!"
echo ""
echo "📁 Available scripts:"
echo "  ./complete_setup.sh       - Run complete setup (recommended)"
echo "  ./setup_project.sh        - Basic project setup"
echo "  ./install_dependencies.sh - Install Python packages"
echo "  ./create_env_files.sh     - Create configuration files"
echo "  ./create_code_templates.sh - Generate code templates"
echo "  ./start_search_first.sh   - Start Search-First system"
echo "  ./start_rag.sh            - Start RAG system"
echo "  ./run_comparison.sh       - Run comparison test"
echo "  ./test_systems.sh         - Quick test both systems"
echo "  ./stop_systems.sh         - Stop all systems"
echo "  ./verify_setup.sh         - Verify complete setup"
echo "  ./clean_project.sh        - Clean up project"
echo ""
echo "🚀 Quick start: ./complete_setup.sh"