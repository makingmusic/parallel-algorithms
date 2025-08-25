# Parallel Algorithms

A Python project for experimenting with algorithms that can operate in parallel, with a focus on sorting algorithms.

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

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Rust extension requirements (optional for Rust sorter)

- Rust toolchain (`rustup`, `cargo`)
- maturin (`pip install maturin`)

## Usage

### Run the benchmark
```bash
python main.py
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

## Requirements

- Python 3.7+
- polars==0.24.0
- torch==2.8.0
- mlx==0.28.0
- numpy==2.3.2
- psutil==7.0.0

### Build the Rust + PyO3 + Rayon sorter

The Rust extension lives in `rust-parallel/` and exposes `rust_parallel_sort()`.

Build and install into your active environment:

```bash
cd rust-parallel
maturin develop --release
```

After this, the Python code can import `rust_parallel` and the algorithm `RUST_PARALLEL_SORT` becomes available.

## Performance

The benchmark will show execution time, memory usage, and correctness verification for each algorithm. POLAR_SORT is particularly effective for large datasets where parallel processing can provide significant performance improvements.
