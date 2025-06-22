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
