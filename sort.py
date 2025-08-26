"""
Sorting Algorithms Module

This module contains implementations of various sorting algorithms with timing wrappers.
Each algorithm has a base implementation and a wrapper function that measures execution time.

Algorithms included:
- Polar Sort (Polars-based parallel sort)
- Rust Parallel Sort (Rayon-based)

Note:
- Basic sorting algorithms (Bubble, Built-in, Quick, Merge, Heap) have been moved to sort_basic_algorithms.py
- MLX/MPS GPU-accelerated sorting algorithms have been moved to sort_mlx.py
"""

import time
import torch
import polars as pl
import logging
from typing import List, Tuple, Any, Dict

# Set up logging for debugging
logger = logging.getLogger(__name__)

# Try importing the Rust extension module if available
try:
    from rust_parallel import rust_parallel_sort as _rust_parallel_sort
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Rust extension not available: {e}")
    _rust_parallel_sort = None
except Exception as e:
    logger.error(f"Unexpected error importing Rust extension: {e}")
    _rust_parallel_sort = None

# Import CPU and memory monitoring utilities
from utils import (
    timing_wrapper_with_monitoring,
    get_available_algorithms as utils_get_available_algorithms,
    get_algorithm as utils_get_algorithm,
    get_display_name as utils_get_display_name,
)

# Import basic sorting algorithms from the dedicated module
try:
    from sort_basic_algorithms import (
        bubble_sort,
        built_in_sort,
        quick_sort,
        merge_sort,
        heap_sort,
        BUBBLE_SORT,
        BUILT_IN_SORT,
        QUICK_SORT,
        MERGE_SORT,
        HEAP_SORT,
    )
except ImportError as e:
    logger.warning(f"Basic sorting algorithms module not available: {e}")
    # Define placeholder functions and constants
    BUBBLE_SORT = "BUBBLE_SORT"
    BUILT_IN_SORT = "BUILT_IN_SORT"
    QUICK_SORT = "QUICK_SORT"
    MERGE_SORT = "MERGE_SORT"
    HEAP_SORT = "HEAP_SORT"

    def bubble_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError("Basic sorting algorithms not available.")

    def built_in_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError("Basic sorting algorithms not available.")

    def quick_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError("Basic sorting algorithms not available.")

    def merge_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError("Basic sorting algorithms not available.")

    def heap_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError("Basic sorting algorithms not available.")

# =============================================================================
# ALGORITHM CONSTANTS
# =============================================================================

# Algorithm name constants (no spaces, easy to use)
POLAR_SORT = "POLAR_SORT"
RUST_PARALLEL_SORT = "RUST_PARALLEL_SORT"

# Display names for user-friendly output
ALGORITHM_DISPLAY_NAMES = {
    POLAR_SORT: "Polar Sort",
    RUST_PARALLEL_SORT: "Rust Parallel Sort (Rayon)",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def timing_wrapper(
    sort_func, arr: List[Any]
) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Generic timing wrapper for sorting functions with CPU and memory monitoring.

    Args:
        sort_func: The sorting function to wrap
        arr: Input array to sort

    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, metrics_dict)
    """
    return timing_wrapper_with_monitoring(sort_func, arr)


# =============================================================================
# ADI SORT
# =============================================================================


def adi_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of Adi sort algorithm.
    """
    return arr


# Basic sorting algorithms have been moved to sort_basic_algorithms.py


# =============================================================================
# MLX/MPS TORCH SORT
# =============================================================================

# Import MLX sorting functions from the dedicated module
try:
    from sort_mlx import (
        mlx_sort,
        mlx_sort_preload_to_memory,
        MLX_SORT,
        MLX_SORT_PRELOAD_TO_MEMORY,
    )
except ImportError as e:
    logger.warning(f"MLX sorting module not available: {e}")
    # Define placeholder functions and constants
    MLX_SORT = "mlx_sort"
    MLX_SORT_PRELOAD_TO_MEMORY = "mlx_sort_preload_to_memory"

    def mlx_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError(
            "MLX sorting not available. Install torch and ensure MPS is available."
        )

    def mlx_sort_preload_to_memory(
        arr: List[Any],
    ) -> Tuple[List[Any], float, Dict[str, float]]:
        raise RuntimeError(
            "MLX sorting not available. Install torch and ensure MPS is available."
        )

# =============================================================================
# POLAR SORT
# =============================================================================


def polar_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of parallel sort using Polars library.

    Polars provides multi-core sorting under the hood using Rust's efficient
    sorting algorithms. This leverages parallel processing capabilities
    for improved performance on large datasets.

    Time Complexity: O(n log n) average case
    Space Complexity: O(n)

    Args:
        arr: Input array to sort

    Returns:
        Sorted array
    """
    # Convert Python list to Polars Series
    series = pl.Series("values", arr)

    # Sort the series (multi-core under the hood)
    sorted_series = series.sort()

    # Convert back to Python list
    return sorted_series.to_list()


def polar_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Polars-based parallel sort with timing measurement.

    Args:
        arr: Input array to sort

    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(polar_sort_impl, arr)


# =============================================================================
# RUST PARALLEL SORT (PyO3 + Rayon)
# =============================================================================


def rust_parallel_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Parallel sort implemented in Rust using Rayon, exposed via PyO3.

    Requires the compiled rust_parallel module to be available.
    """
    if _rust_parallel_sort is None:
        raise RuntimeError(
            "rust_parallel module is not built/available. See README for build instructions."
        )
    # Ensure integers, as Rust signature is Vec<i64>
    return _rust_parallel_sort([int(x) for x in arr])


def rust_parallel_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Rust parallel sort with timing measurement.
    """
    return timing_wrapper(rust_parallel_sort_impl, arr)


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

# Dictionary mapping algorithm names to their wrapper functions
SORTING_ALGORITHMS = {
    POLAR_SORT: polar_sort,
    RUST_PARALLEL_SORT: rust_parallel_sort,
}

# Add basic algorithms if available
try:
    SORTING_ALGORITHMS[BUBBLE_SORT] = bubble_sort
    SORTING_ALGORITHMS[BUILT_IN_SORT] = built_in_sort
    SORTING_ALGORITHMS[QUICK_SORT] = quick_sort
    SORTING_ALGORITHMS[MERGE_SORT] = merge_sort
    SORTING_ALGORITHMS[HEAP_SORT] = heap_sort
    # Add basic algorithm display names
    ALGORITHM_DISPLAY_NAMES[BUBBLE_SORT] = "Bubble Sort"
    ALGORITHM_DISPLAY_NAMES[BUILT_IN_SORT] = "Built-in Sort"
    ALGORITHM_DISPLAY_NAMES[QUICK_SORT] = "Quick Sort"
    ALGORITHM_DISPLAY_NAMES[MERGE_SORT] = "Merge Sort"
    ALGORITHM_DISPLAY_NAMES[HEAP_SORT] = "Heap Sort"
except NameError:
    # Basic algorithms not available
    pass

# Add MLX algorithms if available
try:
    SORTING_ALGORITHMS[MLX_SORT] = mlx_sort
    SORTING_ALGORITHMS[MLX_SORT_PRELOAD_TO_MEMORY] = mlx_sort_preload_to_memory
    # Add MLX display names
    ALGORITHM_DISPLAY_NAMES[MLX_SORT] = "MLX Sort (incl. load)"
    ALGORITHM_DISPLAY_NAMES[MLX_SORT_PRELOAD_TO_MEMORY] = "MLX Sort (preloaded)"
except NameError:
    # MLX algorithms not available
    pass


# Wrapper functions that use the registry utilities from utils.py
def get_available_algorithms() -> List[str]:
    """
    Get list of available sorting algorithm constants.

    Returns:
        List of algorithm constants
    """
    return utils_get_available_algorithms(SORTING_ALGORITHMS, ALGORITHM_DISPLAY_NAMES)


def get_algorithm(name: str):
    """
    Get sorting algorithm function by name.

    Args:
        name: Name of the algorithm (constant or display name)

    Returns:
        Algorithm wrapper function

    Raises:
        KeyError: If algorithm name is not found
    """
    return utils_get_algorithm(name, SORTING_ALGORITHMS, ALGORITHM_DISPLAY_NAMES)


def get_display_name(constant: str) -> str:
    """
    Get display name for an algorithm constant.

    Args:
        constant: Algorithm constant

    Returns:
        Display name for the algorithm
    """
    return utils_get_display_name(constant, ALGORITHM_DISPLAY_NAMES)
