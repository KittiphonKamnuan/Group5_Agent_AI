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
