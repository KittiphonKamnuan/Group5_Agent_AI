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
