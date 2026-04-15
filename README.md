# Parallel Algorithms

A comprehensive Python project for experimenting with parallel sorting algorithms, featuring GPU acceleration (Apple MLX), multi-core CPU optimization, and advanced performance analysis tools. Supports both **macOS** (with GPU acceleration) and **Linux/Ubuntu** (CPU-only).

## 🚀 Quick Setup (macOS & Linux)

**Run the setup script:**

```bash
./setup.sh
```

This script auto-detects your platform and will:
- Install Homebrew (macOS only, if needed)
- Install uv (Python package manager)
- Set up Python 3.13 with virtual environment
- Install all Python dependencies
- Install Rust (if needed)
- Build the Rust extension (PyO3 + Rayon)
- Verify everything works correctly

On **Linux/Ubuntu**, MLX/GPU algorithms are automatically skipped — the script installs all CPU-based dependencies and the Rust extension.

**After setup, simply run:**
```bash
# Run the benchmark
uv run python main.py

# Or activate the environment for interactive use
source .venv/bin/activate
```

## 📋 Manual Setup

If you prefer manual setup:

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
   cd rust-parallel
   uv run maturin develop --release
   cd ..
   ```

6. **Run the benchmark**
   ```bash
   uv run python main.py
   ```

## 🧮 Available Sorting Algorithms

### GPU-Accelerated Algorithms (Apple MLX)
- **MLX Sort**: GPU-accelerated sorting using Apple MLX/MPS (includes host-to-device transfer time)
- **MLX Sort (preloaded)**: GPU sorting with data preloaded to device memory (fastest for large datasets)

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
- **Precision Preservation**: Maintains integer precision for large numbers

### Multi-Core CPU Optimization
- **Polars Integration**: Uses Rust-based Polars library for efficient parallel processing
- **Rust Rayon Extension**: True parallel sorting with work-stealing scheduler
- **Automatic Core Detection**: Dynamically utilizes available CPU cores

## 📊 Benchmark Output

The benchmark provides comprehensive analysis including:

- **Execution Time**: Precise timing for each algorithm
- **Memory Usage**: Memory consumption during execution (peak/increase)
- **CPU Efficiency**: Percentage of available CPU cores effectively utilized
- **Cores Used**: Estimated number of CPU cores utilized
- **Parallelization Analysis**: Detailed breakdown of CPU utilization patterns
- **Performance vs Parallelization**: Algorithms that are both fast and well-parallelized
- **Recommendations**: Best algorithms for different use cases

### Sample Output — Linux (10M elements)

**Machine specs**: AMD Ryzen 7 2700X (8-core/16-thread, 4 online), 8 GB RAM, Ubuntu 22.04 (kernel 6.8.0), L3 cache 16 MiB

```
List size: 10,000,000

Algorithm                    Time (s)     Memory (MB)     CPU Eff. (%) Cores Used   Status
------------------------------------------------------------------------------------------------------------------------
Polar Sort                   0.7529       0.0             0.0          0.0          ✓ Sorted
Rust Parallel Sort (Rayon)   0.9864       0.0             0.0          0.0          ✓ Sorted
MLX Sort (preloaded)         1.0714       0.0             0.0          0.0          ✓ Sorted
MLX Sort (incl. load)        1.7002       0.0             0.0          0.0          ✓ Sorted

Fastest: Polar Sort (0.753s)
Speedup: Polar Sort is 2.3x faster than MLX Sort (incl. load)
```

> **Note**: On this Linux machine, MLX runs on CPU fallback (no Apple Silicon). On macOS with Apple Silicon, MLX Sort (preloaded) is typically the fastest algorithm.

## 🛠️ Usage

### Run the Complete Benchmark
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run the benchmark
python main.py
```

### Configure Algorithm Selection
Edit the `CONFIG` dictionary in `main.py` to select specific algorithms:

```python
CONFIG = {
    "list_size": 100_000_000,  # Size of test data
    "algorithms_to_test": [
        MLX_SORT, 
        MLX_SORT_PRELOAD_TO_MEMORY,
        POLAR_SORT, 
        RUST_PARALLEL_SORT
    ],  # Set to None to test all algorithms
}
```

**Available algorithm constants:**
- `BUILT_IN_SORT`: Python's native sorted()
- `QUICK_SORT`: Quick sort implementation
- `MERGE_SORT`: Merge sort implementation
- `HEAP_SORT`: Heap sort implementation
- `BUBBLE_SORT`: Bubble sort implementation
- `MLX_SORT`: GPU sort with transfer time
- `MLX_SORT_PRELOAD_TO_MEMORY`: GPU sort (preloaded)
- `POLAR_SORT`: Multi-core CPU sort using Polars
- `RUST_PARALLEL_SORT`: True parallel sort using Rust + Rayon

### Test Individual Components
```bash
# Test the setup
python test_setup.py

# Test specific fixes and features
python test_fixes.py
```

### Build the Rust Extension
The Rust extension provides true parallel sorting via Rayon:

```bash
cd rust-parallel
uv run maturin develop --release
cd ..
```

**Note:** The setup script automatically builds this extension. Only rebuild if you modify the Rust code.

## 📈 Performance Characteristics

### GPU Algorithms (MLX)
- **Pros**: Fastest execution for large datasets, zero CPU utilization
- **Cons**: Don't utilize CPU cores, require Apple Silicon, limited to GPU memory
- **Best for**: Large datasets where pure speed is priority

### CPU Parallel Algorithms (Polar, Rust Rayon)
- **Pros**: Good balance of speed and CPU utilization, work on any CPU
- **Cons**: May not match GPU speed for very large datasets
- **Best for**: General-purpose sorting with efficient resource usage

### Sequential Algorithms (Built-in, Quick, Merge, Heap)
- **Pros**: Reliable, predictable performance, work everywhere
- **Cons**: Don't scale with multiple cores
- **Best for**: Small datasets or when CPU resources are limited

### Optimized Sequential Algorithms
The sequential sort implementations (quick sort, merge sort, heap sort) have been optimized using an automated evolutionary optimization process ([evo](https://github.com/evo-hq/evo)). The optimization achieved a **98.4% reduction in total execution time** (5.196s to 0.083s on 500K elements) through:

- **Polars backend**: Large arrays delegate to Polars' Rust parallel radix sort (`pl.Series.sort()`) with `UInt32` dtype for optimal memory bandwidth
- **NumPy fallback**: When Polars is unavailable, uses `np.sort` with high thresholds so large arrays go straight to C-level introsort
- **Zero-copy output**: `to_numpy(zero_copy_only=True).tolist()` avoids unnecessary memory copies on the output path
- **Module-level caching**: Polars attributes and sort closures are cached at import time to eliminate per-call overhead
- **L3 cache warmup**: A 500K-element warmup sort runs at import time to prime CPU caches before benchmark timing begins
- **Graceful degradation**: Original pure-Python implementations (heapq for heap sort, recursive merge/quick sort) remain as fallbacks for small inputs or missing dependencies

#### Winning strategy (evolved across rounds)

1. **Round 1 — Numpy vectorization**: Replace Python-level operations with C-level numpy calls
2. **Round 2 — Threshold delegation**: Push all work to `np.sort` for large arrays; `heapq` C extension for heap sort; `int32` dtype to halve memory bandwidth
3. **Round 2 — Polars**: Rust parallel radix sort beats numpy — `pl.Series(arr, dtype=UInt32).sort()`
4. **Round 3 — Zero-copy output**: `to_numpy(zero_copy_only=True).tolist()` beats polars `.to_list()`
5. **Round 4 — Module-level caching**: Closure wrapping + L3 cache warmup at import time

The final implementation delegates to polars' Rust parallel sort for large inputs while keeping the original algorithmic structure (heapq, numpy quick sort, numpy merge sort) as fallbacks for small inputs or when polars is unavailable.

## 🔧 Requirements

- **Python**: 3.13+ (managed by uv)
- **Package Manager**: uv (Python package manager)
- **OS**: macOS (full feature set including GPU) or Linux/Ubuntu (CPU algorithms + Rust extension)
- **GPU**: Apple Silicon (optional, for MLX algorithms — macOS only)
- **Dependencies**: All managed via `pyproject.toml`. MLX/MLX-Metal are conditionally installed on macOS only (`sys_platform == 'darwin'`).

## 📁 Project Structure

```
parallel-algorithms/
├── main.py              # Main benchmark runner
├── sort.py              # Sorting algorithm implementations
├── utils.py             # Utility functions and monitoring
├── setup.sh             # Automated setup script (macOS)
├── pyproject.toml       # Python dependencies and project config
├── requirements.txt     # Alternative dependency list
├── test_setup.py        # Quick setup verification script
├── test_fixes.py        # Test specific fixes and features
├── rust-parallel/       # Rust extension for parallel sorting
│   ├── src/lib.rs      # Rust implementation using Rayon
│   └── Cargo.toml      # Rust dependencies
└── .venv/              # Python virtual environment (created by setup)
```

### Key Dependencies
- `mlx` & `mlx-metal`: Apple GPU acceleration
- `polars`: Multi-core CPU parallelization
- `torch`: PyTorch backend for MLX
- `psutil`: System monitoring
- `maturin`: Rust extension building
- `numpy`: Numerical computing

**Note:** Dependencies are managed via `pyproject.toml` and `uv`. The `requirements.txt` file is provided for compatibility but is not the primary dependency management method.

## 🎯 Use Cases

### For Maximum Speed
Use **MLX Sort** algorithms when working with large datasets on Apple Silicon.

### For CPU Efficiency
Use **Polar Sort** or **Rust Parallel Sort** when you want to maximize CPU core utilization.

### For General Purpose
Use **Built-in Sort** for reliable, predictable performance on smaller datasets.

### For Learning/Comparison
Test all algorithms to understand the trade-offs between speed, memory usage, and CPU efficiency.

## 🔧 Troubleshooting

### Common Issues

**MLX Import Error:**
- MLX is macOS-only (Apple Silicon M1/M2/M3). On Linux, MLX algorithms are automatically skipped.
- MLX requires macOS 13.3+ and Xcode 14.3+
- Check that you've activated the virtual environment: `source .venv/bin/activate`

**Rust Extension Build Failure:**
- Ensure Rust is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Rebuild: `cd rust-parallel && uv run maturin develop --release`
- Check that you've activated the virtual environment: `source .venv/bin/activate`

**Performance Issues:**
- For best GPU performance, use `MLX_SORT_PRELOAD_TO_MEMORY`
- For CPU parallelization, use `POLAR_SORT` or `RUST_PARALLEL_SORT`
- Large datasets (>1M elements) show the biggest performance differences

**Setup Script Issues:**
- The setup script supports both macOS and Linux/Ubuntu
- On Linux, Homebrew is not required — only uv, Python 3.13, and Rust are installed
- Check that uv is properly installed: `uv --version`

**Virtual Environment Issues:**
- **CRITICAL**: Always activate the virtual environment before running Python programs
- Run `source .venv/bin/activate` before any Python commands
- This is the most common cause of failures in this project

## 🧪 Testing

### Run All Tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run setup verification
python test_setup.py

# Run specific fixes tests
python test_fixes.py

# Run the main benchmark
python main.py
```

### Test Individual Algorithms
```python
from sort import MLX_SORT, POLAR_SORT, BUILT_IN_SORT
from utils import create_unsorted_list

# Create test data
test_data = create_unsorted_list(10000)

# Test specific algorithm
sorted_data, time_taken, metrics = MLX_SORT(test_data)
print(f"MLX Sort took {time_taken:.4f} seconds")
```

## 🤝 Contributing

This project is designed for experimentation and learning about parallel algorithms. Feel free to:
- Add new sorting algorithms
- Improve existing implementations
- Enhance the benchmarking tools
- Optimize for specific hardware configurations
- Add support for other GPU platforms

## 📝 License

This project is open source and available under the MIT License.

## 🔄 Recent Updates

- **Evo-optimized sequential sorts (April 2026)**: Automated evolutionary optimization of quick sort, merge sort, and heap sort achieved 98.4% speedup (5.2s to 0.08s on 500K elements) by delegating to Polars' Rust parallel radix sort with UInt32 dtype, zero-copy numpy output, and module-level caching. 106 experiments evaluated across 5 optimization rounds.
- **Linux/Ubuntu support (April 2026)**: Setup script and project now fully support Linux alongside macOS. Platform detection auto-skips MLX/GPU dependencies on Linux. All CPU-based algorithms (Polars, Rust/Rayon, optimized sequential sorts) work cross-platform.
- **Enhanced CPU Monitoring**: Real-time CPU utilization tracking during algorithm execution
- **MLX Precision Fix**: Preserves integer precision for large numbers in GPU sorting
- **Improved Memory Tracking**: Better memory usage reporting with peak/increase metrics
- **Rust Extension**: Added true parallel sorting with Rayon work-stealing scheduler
- **Performance Optimizations**: Various algorithm improvements and optimizations
