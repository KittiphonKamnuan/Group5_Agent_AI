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
