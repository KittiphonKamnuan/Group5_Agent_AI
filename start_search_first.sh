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
