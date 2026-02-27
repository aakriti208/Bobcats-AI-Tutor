#!/bin/bash
# SimpleRAG Installation Script for Unix/Mac/Linux
# This script automates the installation of SimpleRAG for Canvas LMS

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Function to display usage
show_usage() {
    echo "=========================================="
    echo "SimpleRAG Installation Script"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  $0 --token <API_TOKEN> --courses <COURSE_IDS>"
    echo ""
    echo "Required Arguments:"
    echo "  --token <API_TOKEN>      Your Canvas API token"
    echo "  --courses <COURSE_IDS>   Comma-separated course IDs (e.g., 12345,67890)"
    echo ""
    echo "Options:"
    echo "  --help                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --token abc123xyz --courses 12345,67890"
    echo ""
    exit 1
}

# Parse command-line arguments
CANVAS_API_TOKEN=""
CANVAS_COURSE_IDS=""
CANVAS_BASE_URL="https://canvas.yourinstitution.edu/"

while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            CANVAS_API_TOKEN="$2"
            shift 2
            ;;
        --courses)
            CANVAS_COURSE_IDS="$2"
            shift 2
            ;;
        --help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$CANVAS_API_TOKEN" ] || [ -z "$CANVAS_COURSE_IDS" ]; then
    print_error "Missing required arguments"
    echo ""
    show_usage
fi

echo "=========================================="
echo "SimpleRAG Installation Script"
echo "=========================================="
echo ""
print_info "Configuration:"
echo "  Canvas URL: $CANVAS_BASE_URL"
echo "  Course IDs: $CANVAS_COURSE_IDS"
echo "  API Token: ${CANVAS_API_TOKEN:0:10}..." # Show only first 10 chars for security
echo ""

# Check Python version
print_info "Checking Python installation..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
        PYTHON_CMD=python3
    else
        print_error "Python 3.11+ is required. Found: $PYTHON_VERSION"
        echo "Please install Python 3.11 or newer from https://www.python.org/"
        exit 1
    fi
else
    print_error "Python 3 is not installed."
    echo "Please install Python 3.11+ from https://www.python.org/"
    exit 1
fi
print_success "Python found: $($PYTHON_CMD --version)"

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_info "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi
$PYTHON_CMD -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"

# Install requirements
print_info "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt
print_success "Python dependencies installed"

# Setup environment file
print_info "Setting up environment configuration..."
if [ -f ".env" ]; then
    print_info "Backing up existing .env file to .env.backup"
    cp .env .env.backup
fi

# Create .env file with provided configuration
cat > .env << EOF
# Canvas API Configuration
CANVAS_API_TOKEN=$CANVAS_API_TOKEN
CANVAS_BASE_URL=$CANVAS_BASE_URL
CANVAS_COURSE_IDS=$CANVAS_COURSE_IDS

# Optional: Override default settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=gemma:2b
TOP_K_RESULTS=3
SIMILARITY_THRESHOLD=0.5

# Ingestion Settings
INGEST_BATCH_SIZE=100
INGEST_MAX_WORKERS=4
INGEST_CONTENT_TYPES=module,page,assignment,announcement,discussion,file
PDF_EXTRACTION_TIMEOUT=30
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO
EOF

print_success "Created .env file with your configuration"

# Check for Ollama
print_info "Checking for Ollama installation..."
if command -v ollama &> /dev/null; then
    print_success "Ollama is installed"

    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service is running"

        # Check if model is installed
        print_info "Checking for gemma:2b model..."
        if ollama list | grep -q "gemma:2b"; then
            print_success "gemma:2b model is installed"
        else
            print_info "Installing gemma:2b model (this will take a few minutes)..."
            ollama pull gemma:2b
            print_success "gemma:2b model installed"
        fi
    else
        print_info "Ollama is installed but not running"
        print_info "Start it with: ollama serve"
    fi
else
    print_error "Ollama is not installed"
    echo ""
    echo "Ollama is required for answer generation."
    echo "Installation instructions:"
    echo "  macOS/Linux: Visit https://ollama.ai and download the installer"
    echo "  After installation, run:"
    echo "    ollama serve"
    echo "    ollama pull gemma:2b"
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data/chroma_db
mkdir -p data/metadata
print_success "Directories created"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Your .env file has been configured with:"
echo "  Canvas URL: $CANVAS_BASE_URL"
echo "  Course IDs: $CANVAS_COURSE_IDS"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (in new terminal sessions):"
echo "   source venv/bin/activate"
echo ""
echo "2. Ingest your Canvas content:"
echo "   python scripts/ingest_data.py --course YOUR_COURSE_ID --full"
echo "   (Replace YOUR_COURSE_ID with one of: $CANVAS_COURSE_IDS)"
echo ""
echo "3. Start the web interface:"
echo "   python app.py"
echo "   Then visit: http://localhost:8000"
echo ""
echo "Note: To modify settings, edit the .env file in the project root"
echo ""
echo "For help, see README.md or contact support"
echo ""
