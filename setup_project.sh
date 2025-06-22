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
