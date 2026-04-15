# Parallel Algorithms Benchmarking Suite

A Python benchmarking project that compares parallel sorting algorithms across GPU (Apple MLX/Metal), multi-core CPU (Polars, Rust/Rayon), and sequential baselines.

## Project Structure

```
main.py                    # Entry point — CONFIG dict controls list size & which algorithms to run
sort.py                    # Algorithm registry (SORTING_ALGORITHMS dict) & constants
sort_basic_algorithms.py   # Sequential sorts: built-in, quick, merge, heap, bubble
sort_mlx.py                # GPU-accelerated MLX sorts (with/without transfer time)
utils.py                   # Timing wrapper, CPU/memory monitoring (thread-safe via Queue), helpers
rust-parallel/             # PyO3 + Rayon Rust extension for parallel sort
  src/lib.rs
  Cargo.toml
```

## Setup & Running

```bash
# macOS automated setup
./setup.sh
uv run python main.py

# Manual (all platforms)
uv sync --python 3.13
cd rust-parallel && uv run maturin develop --release && cd ..
uv run python main.py
```

Package manager: **uv**. Python 3.13+.

## Key Dependencies

- `mlx` / `mlx-metal` — Apple Silicon GPU acceleration
- `polars` — Rust-backed parallel sort
- `maturin` — builds the Rust FFI extension
- `psutil` — CPU/memory monitoring
- `numpy`, `torch` — numerical ops

## Architecture Notes

- **Registry pattern**: algorithms register in `SORTING_ALGORITHMS` dict in `sort.py`. Add new algorithms there.
- **Timing/monitoring wrapper**: every algorithm runs through `timing_wrapper_with_monitoring()` in `utils.py`, which spawns background threads for CPU and memory sampling.
- **Graceful degradation**: missing optional deps (MLX on non-Apple, unbuilt Rust extension) log warnings instead of crashing.
- **MLX two-variant design**: `MLX_SORT` includes host-to-GPU transfer time; `MLX_SORT_PRELOAD_TO_MEMORY` excludes it. Both exist to separate transfer overhead from compute.

## Testing

```bash
python test_setup.py    # Verify algorithm implementations
python test_fixes.py    # Feature-specific tests
```

## Conventions

- Algorithm constants are defined in `sort.py` and imported everywhere by name (e.g. `BUILT_IN_SORT`, `RUST_PARALLEL_SORT`).
- Display names are separate from constants — use `get_display_name()`.
- Benchmark config lives in `CONFIG` dict at the top of `main.py`.
