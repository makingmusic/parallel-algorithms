# Parallel Algorithms

A comprehensive Python project for experimenting with parallel sorting algorithms, featuring GPU acceleration, multi-core CPU utilization, and performance analysis tools.

## 🚀 Quick Setup (macOS)

**For a fresh MacBook, run this single command:**

```bash
./setup.sh
```

This script will automatically:
- Install Homebrew (if needed)
- Install uv (Python package manager)
- Set up Python 3.13 with virtual environment
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

## 🧮 Available Sorting Algorithms

### GPU-Accelerated Algorithms
- **MLX Sort**: GPU-accelerated sorting using Apple MLX/MPS
- **MLX Sort (preloaded)**: GPU sorting with data preloaded to device memory

### Multi-Core CPU Algorithms
- **Polar Sort**: Multi-core parallel sorting using Polars library
- **Rust Parallel Sort (Rayon)**: True parallel sort via Rust + PyO3 + Rayon

### Sequential Algorithms
- **Built-in Sort**: Python's native `sorted()` function
- **Quick Sort**: Efficient divide-and-conquer algorithm
- **Merge Sort**: Stable divide-and-conquer algorithm
- **Heap Sort**: In-place sorting using heap data structure
- **Bubble Sort**: Simple O(n²) sorting algorithm

## ✨ Key Features

### Advanced Performance Analysis
- **Real-time CPU Monitoring**: Tracks CPU utilization during algorithm execution
- **Parallelization Efficiency**: Measures how effectively algorithms use multiple cores
- **Memory Usage Tracking**: Monitors memory consumption for each algorithm
- **Performance vs Parallelization Analysis**: Identifies algorithms that are both fast and efficient

### GPU Acceleration
- **Apple MLX Integration**: Leverages Apple Silicon GPU for sorting operations
- **Host-to-Device Transfer Optimization**: Includes transfer time in performance metrics
- **Preloaded Data Support**: Option to preload data to GPU memory for faster execution

### Multi-Core CPU Optimization
- **Polars Integration**: Uses Rust-based Polars library for efficient parallel processing
- **Rust Rayon Extension**: True parallel sorting with work-stealing scheduler
- **Automatic Core Detection**: Dynamically utilizes available CPU cores

## 📊 Benchmark Output

The benchmark provides comprehensive analysis including:

- **Execution Time**: Precise timing for each algorithm
- **Memory Usage**: Memory consumption during execution
- **CPU Efficiency**: Percentage of available CPU cores effectively utilized
- **Cores Used**: Estimated number of CPU cores utilized
- **Parallelization Analysis**: Detailed breakdown of CPU utilization patterns
- **Performance vs Parallelization**: Algorithms that are both fast and well-parallelized
- **Recommendations**: Best algorithms for different use cases

### Sample Output
```
Algorithm                    Time (s)  Memory (MB)  CPU Efficiency  Cores Used
MLX Sort (preloaded)          0.045       45.2      GPU/Other        N/A
Polar Sort                    0.234       67.8      85.2% efficiency  6.8 cores
Rust Parallel Sort (Rayon)    0.312       52.1      78.9% efficiency  6.3 cores
Built-in Sort                 1.234       23.4      12.3% efficiency  1.0 cores
```

## 🛠️ Usage

### Run the Complete Benchmark
```bash
uv run python main.py
```

### Configure Algorithm Selection
Edit the `CONFIG` dictionary in `main.py` to select specific algorithms:

```python
CONFIG = {
    "list_size": 10_000_000,
    "algorithms_to_test": [
        BUILT_IN_SORT, 
        MLX_SORT, 
        POLAR_SORT, 
        RUST_PARALLEL_SORT
    ],  # Set to None to test all algorithms
}
```

### Build the Rust Extension
The Rust extension provides true parallel sorting via Rayon:

```bash
cd rust-parallel
maturin develop --release
```

## 📈 Performance Characteristics

### GPU Algorithms (MLX)
- **Pros**: Fastest execution for large datasets
- **Cons**: Don't utilize CPU cores, require Apple Silicon
- **Best for**: Large datasets where pure speed is priority

### CPU Parallel Algorithms (Polar, Rust Rayon)
- **Pros**: Good balance of speed and CPU utilization
- **Cons**: May not match GPU speed for very large datasets
- **Best for**: General-purpose sorting with efficient resource usage

### Sequential Algorithms (Built-in, Quick, Merge, Heap)
- **Pros**: Reliable, predictable performance
- **Cons**: Don't scale with multiple cores
- **Best for**: Small datasets or when CPU resources are limited

## 🔧 Requirements

- **Python**: 3.13+ (managed by uv)
- **Package Manager**: uv (Python package manager)
- **GPU**: Apple Silicon (for MLX algorithms)
- **Dependencies**: All managed via `pyproject.toml`

### Key Dependencies
- `mlx` & `mlx-metal`: Apple GPU acceleration
- `polars`: Multi-core CPU parallelization
- `torch`: PyTorch backend for MLX
- `psutil`: System monitoring
- `maturin`: Rust extension building

## 🎯 Use Cases

### For Maximum Speed
Use **MLX Sort** algorithms when working with large datasets on Apple Silicon.

### For CPU Efficiency
Use **Polar Sort** or **Rust Parallel Sort** when you want to maximize CPU core utilization.

### For General Purpose
Use **Built-in Sort** for reliable, predictable performance on smaller datasets.

### For Learning/Comparison
Test all algorithms to understand the trade-offs between speed, memory usage, and CPU efficiency.

## 🤝 Contributing

This project is designed for experimentation and learning about parallel algorithms. Feel free to:
- Add new sorting algorithms
- Improve existing implementations
- Enhance the benchmarking tools
- Optimize for specific hardware configurations

## 📝 License

This project is open source and available under the MIT License.
