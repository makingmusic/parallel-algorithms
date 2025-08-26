#!/bin/bash

# Parallel Algorithms Setup Script for macOS
# This script sets up the entire project environment on a fresh MacBook

set -e  # Exit on any error

echo "🚀 Setting up Parallel Algorithms project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_note() {
    echo -e "${CYAN}[NOTE]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please use the manual setup instructions in README.md for other operating systems."
    exit 1
fi

print_step "Checking macOS version and architecture..."
# Check macOS version
macos_version=$(sw_vers -productVersion)
print_status "macOS version: $macos_version"

# Check architecture
arch=$(uname -m)
print_status "Architecture: $arch"

if [[ "$arch" != "arm64" ]]; then
    print_warning "This project is optimized for Apple Silicon (M1/M2/M3). Performance may be limited on Intel Macs."
fi

# Check if Homebrew is installed
print_step "Checking Homebrew installation..."
if ! command_exists brew; then
    print_status "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for this session
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        export PATH="/opt/homebrew/bin:$PATH"
        print_status "Added Homebrew to PATH (/opt/homebrew/bin)"
    elif [[ -f "/usr/local/bin/brew" ]]; then
        export PATH="/usr/local/bin:$PATH"
        print_status "Added Homebrew to PATH (/usr/local/bin)"
    fi
    print_success "Homebrew installed successfully"
else
    print_success "Homebrew already installed"
    # Update Homebrew
    print_status "Updating Homebrew..."
    brew update >/dev/null 2>&1 || print_warning "Homebrew update failed (continuing anyway)"
fi

# Check if uv is installed
print_step "Checking uv installation..."
if ! command_exists uv; then
    print_status "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env  # uv is installed via cargo
    print_success "uv installed successfully"
else
    print_success "uv already installed"
    # Update uv
    print_status "Updating uv..."
    uv self update >/dev/null 2>&1 || print_warning "uv update failed (continuing anyway)"
fi

# Use uv to ensure Python 3.13
print_step "Setting up Python 3.13 with uv..."
uv python install 3.13
print_success "Python 3.13 installed via uv"

# Set up uv project
print_step "Setting up uv project..."
uv sync --python 3.13
print_success "uv project configured"

# Verify Python version
python_version=$(uv run python --version)
print_success "Using $python_version (managed by uv)"

# Create virtual environment and install dependencies with uv
print_step "Creating virtual environment and installing dependencies..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf .venv
fi

# Install dependencies using uv
print_status "Installing Python dependencies..."
uv sync
print_success "Virtual environment created and dependencies installed with uv"

# Check if Rust is installed
print_step "Checking Rust installation..."
if ! command_exists rustc; then
    print_status "Rust not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    print_success "Rust installed successfully"
else
    print_success "Rust already installed"
    # Update Rust
    print_status "Updating Rust..."
    rustup update >/dev/null 2>&1 || print_warning "Rust update failed (continuing anyway)"
fi

# Install maturin for building Rust extensions
print_step "Installing maturin..."
uv add maturin

# Build the Rust extension
print_step "Building Rust extension..."
cd rust-parallel
print_status "Building Rust extension with maturin..."
uv run maturin develop --release
cd ..
print_success "Rust extension built successfully"

# Verify installation
print_step "Verifying installation..."
print_status "Testing Python imports..."
uv run python -c "
import sys
import polars
import torch
import mlx
import numpy
import psutil
print('✅ All Python dependencies imported successfully')

try:
    import rust_parallel
    print('✅ Rust extension imported successfully')
except ImportError as e:
    print(f'❌ Rust extension import failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Installation verification completed"
else
    print_error "Installation verification failed"
    exit 1
fi

# Create a simple test script
print_step "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import time
import random
from sort import POLAR_SORT, MLX_SORT, BUILT_IN_SORT

def test_setup():
    print("🧪 Testing setup...")
    
    # Create a small test list
    test_list = [random.randint(1, 1000) for _ in range(1000)]
    
    # Test a few algorithms
    algorithms = [
        ("Built-in Sort", BUILT_IN_SORT),
        ("Polar Sort", POLAR_SORT),
        ("MLX Sort", MLX_SORT),
    ]
    
    for name, algorithm in algorithms:
        try:
            start_time = time.time()
            sorted_list, execution_time, cpu_metrics = algorithm(test_list)
            end_time = time.time()
            
            # Verify sorting
            is_sorted = all(sorted_list[i] <= sorted_list[i + 1] for i in range(len(sorted_list) - 1))
            
            if is_sorted:
                print(f"✅ {name}: {execution_time:.4f}s")
            else:
                print(f"❌ {name}: Failed to sort correctly")
                
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("🎉 Setup test completed!")

if __name__ == "__main__":
    test_setup()
EOF

print_success "Test script created"

# Show installed versions
print_step "Verifying installed versions..."
echo ""
echo "📋 Installed Versions:"
echo "======================"

# Python and uv versions
echo "🐍 Python: $python_version"
echo "📦 uv: $(uv --version)"

# Rust version
if command_exists rustc; then
    echo "🦀 Rust: $(rustc --version | cut -d' ' -f2)"
    echo "📦 Cargo: $(cargo --version | cut -d' ' -f2)"
else
    echo "❌ Rust: Not found"
fi

# Key Python packages
echo ""
echo "📚 Key Python Packages:"
echo "----------------------"
uv run python -c "
import sys
import polars
import torch
import mlx
import numpy
import psutil
import maturin

print(f'polars: {polars.__version__}')
print(f'torch: {torch.__version__}')
print(f'mlx: {getattr(mlx, \"__version__\", \"unknown\")}')
print(f'numpy: {numpy.__version__}')
print(f'psutil: {psutil.__version__}')
print(f'maturin: {getattr(maturin, \"__version__\", \"unknown\")}')
"

# Check if Rust extension is working
echo ""
echo "🔧 Rust Extension:"
echo "-----------------"
uv run python -c "
try:
    import rust_parallel
    print('✅ rust_parallel: Imported successfully')
except ImportError as e:
    print(f'❌ rust_parallel: {e}')
"

# Virtual environment info
echo ""
echo "🏠 Virtual Environment:"
echo "---------------------"
echo "Location: $(pwd)/.venv"
echo "Python: $(uv run which python)"
echo "uv: $(which uv)"

# Performance recommendations
echo ""
echo "🚀 Performance Recommendations:"
echo "-----------------------------"
if [[ "$arch" == "arm64" ]]; then
    echo "✅ Apple Silicon detected - GPU acceleration available"
    echo "💡 For best performance, use MLX_SORT_PRELOAD_TO_MEMORY"
else
    echo "⚠️  Intel Mac detected - GPU acceleration not available"
    echo "💡 For best performance, use POLAR_SORT or RUST_PARALLEL_SORT"
fi

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Next Steps:"
echo "=============="
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the benchmark:"
echo "   python main.py"
echo ""
echo "3. Test the setup:"
echo "   python test_setup.py"
echo ""
echo "4. Test specific features:"
echo "   python test_fixes.py"
echo ""
print_note "Remember to always activate the virtual environment before running Python programs!"
echo ""
echo "📚 For more information, see README.md"
echo ""
