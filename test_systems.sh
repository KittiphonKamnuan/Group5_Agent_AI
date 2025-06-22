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
