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
