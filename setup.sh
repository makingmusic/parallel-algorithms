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

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please use the manual setup instructions in README.md for other operating systems."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_status "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for this session
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        export PATH="/opt/homebrew/bin:$PATH"
    elif [[ -f "/usr/local/bin/brew" ]]; then
        export PATH="/usr/local/bin:$PATH"
    fi
    print_success "Homebrew installed successfully"
else
    print_success "Homebrew already installed"
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_status "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env  # uv is installed via cargo
    print_success "uv installed successfully"
else
    print_success "uv already installed"
fi

# Use uv to ensure Python 3.13
print_status "Setting up Python 3.13 with uv..."
uv python install 3.13
print_success "Python 3.13 installed via uv"

# Set up uv project
print_status "Setting up uv project..."
uv sync --python 3.13
print_success "uv project configured"

# Verify Python version
print_success "Using Python 3.13 (managed by uv)"

# Create virtual environment and install dependencies with uv
print_status "Creating virtual environment and installing dependencies with uv..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf .venv
fi

# Install dependencies using uv
uv sync
print_success "Virtual environment created and dependencies installed with uv"

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    print_status "Rust not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    print_success "Rust installed successfully"
else
    print_success "Rust already installed"
fi

# Install maturin for building Rust extensions
print_status "Installing maturin..."
uv add maturin

# Build the Rust extension
print_status "Building Rust extension..."
cd rust-parallel
uv run maturin develop --release
cd ..
print_success "Rust extension built successfully"

# Verify installation
print_status "Verifying installation..."
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
print_status "Creating test script..."
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
print_status "Verifying installed versions..."
echo ""
echo "📋 Installed Versions:"
echo "======================"

# Python and uv versions
echo "🐍 Python: 3.13 (managed by uv)"
echo "📦 uv: $(uv --version)"

# Rust version
if command -v rustc &> /dev/null; then
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
echo "Python: $(which python3)"
echo "uv: $(which uv)"

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Run the benchmark: uv run python main.py"
echo "2. Test the setup: uv run python test_setup.py"
echo "3. Or activate the environment: source .venv/bin/activate"
echo ""
echo "For more information, see README.md"
echo ""
