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
