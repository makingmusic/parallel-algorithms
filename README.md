# Parallel Algorithms

A Python project for experimenting with algorithms that can operate in parallel, with a focus on sorting algorithms.

## 🚀 Quick Setup (macOS)

**For a fresh MacBook, run this single command:**

```bash
./setup.sh
```

This script will automatically:
- Install Homebrew (if needed)
- Create a Python virtual environment
- Install all Python dependencies
- Install Rust (if needed)
- Build the Rust extension
- Verify everything works correctly

**After setup, simply run:**
```bash
uv run python main.py      # Run the benchmark
# Or activate the environment: source .venv/bin/activate
```

## 📋 Manual Setup (Other Platforms)

If you're not on macOS or prefer manual setup:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd parallel-algorithms
   ```

2. **Install uv (Python package manager)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.cargo/env
   ```

3. **Set up Python 3.13 and install dependencies**
   ```bash
   uv sync --python 3.13
   ```

4. **Install Rust (for Rust extension)**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

5. **Build Rust extension**
   ```bash
   pip install maturin
   cd rust-parallel
   maturin develop --release
   cd ..
   ```

6. **Run the benchmark**
   ```bash
   uv run python main.py
   ```

## Available Sorting Algorithms

- **Built-in Sort**: Python's native `sorted()` function
- **Bubble Sort**: Simple O(n²) sorting algorithm
- **Quick Sort**: Efficient divide-and-conquer algorithm
- **Merge Sort**: Stable divide-and-conquer algorithm
- **Heap Sort**: In-place sorting using heap data structure
- **MLX Sort**: GPU-accelerated sorting using PyTorch MLX/MPS
- **MLX Sort (preloaded)**: GPU sorting with preloaded data
- **Polar Sort**: Multi-core parallel sorting using Polars library
- **Rust Parallel Sort (Rayon)**: True parallel sort via Rust + PyO3

## Features

- **POLAR_SORT**: A new parallel sorting algorithm that leverages the Polars library
  - Uses Rust's efficient sorting algorithms under the hood
  - Automatically utilizes multiple CPU cores for parallel processing
  - Time Complexity: O(n log n) average case
  - Space Complexity: O(n)

- **CPU Parallelization Monitoring**: Real-time CPU utilization tracking
  - Measures parallelization efficiency for each algorithm
  - Shows effective number of CPU cores utilized
  - Provides detailed performance vs parallelization analysis
  - Helps identify the most efficient algorithms for your hardware

## Usage

### Run the benchmark
```bash
uv run python main.py
```

### Test POLAR_SORT specifically
```bash
python test_polar_sort.py
```

### Configure algorithms to test
Edit the `CONFIG` dictionary in `main.py` to select which algorithms to benchmark:

```python
CONFIG = {
    "list_size": 10_000,
    "algorithms_to_test": [BUILT_IN_SORT, POLAR_SORT, MLX_SORT],  # Test specific algorithms
    # or set to None to test all algorithms
}
```

### Build the Rust + PyO3 + Rayon sorter

The Rust extension lives in `rust-parallel/` and exposes `rust_parallel_sort()`.

Build and install into your active environment:

```bash
cd rust-parallel
maturin develop --release
```

After this, the Python code can import `rust_parallel` and the algorithm `RUST_PARALLEL_SORT` becomes available.

## Benchmark Output

The benchmark provides comprehensive analysis including:

- **Execution Time**: How fast each algorithm completes
- **Memory Usage**: Memory consumption during execution
- **CPU Efficiency**: Percentage of available CPU cores effectively utilized
- **Cores Used**: Estimated number of CPU cores utilized
- **Parallelization Analysis**: Detailed breakdown of CPU utilization patterns
- **Performance vs Parallelization**: Algorithms that are both fast and well-parallelized
- **Recommendations**: Best algorithms for different use cases (speed vs efficiency)

## Requirements

- Python 3.13 (managed by uv)
- uv (Python package manager)
- All dependencies are automatically managed via pyproject.toml

## Performance

The benchmark will show execution time, memory usage, CPU parallelization efficiency, and correctness verification for each algorithm. 

- **GPU algorithms** (MLX) typically provide the fastest execution but don't utilize CPU cores
- **CPU parallel algorithms** (Polar, Rust Rayon) provide good balance of speed and CPU utilization
- **Sequential algorithms** (Built-in, Quick, Merge, Heap) are reliable but may not scale with multiple cores

The parallelization analysis helps you choose the right algorithm based on your specific needs: pure speed, CPU efficiency, or a balance of both.
