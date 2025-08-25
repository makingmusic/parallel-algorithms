"""
Sorting Algorithms Module

This module contains implementations of various sorting algorithms with timing wrappers.
Each algorithm has a base implementation and a wrapper function that measures execution time.

Algorithms included:
- Bubble Sort
- Built-in Sort (Python's sorted())
- Quick Sort
- Merge Sort
- Heap Sort
- MLX Sort (PyTorch MPS/MLX, includes host-to-device transfer)
- MLX Sort (PyTorch MPS/MLX, preloaded to device)
"""

import time
import torch
import polars as pl
from typing import List, Tuple, Any, Dict

# Try importing the Rust extension module if available
try:
    from rust_parallel import rust_parallel_sort as _rust_parallel_sort
except Exception:
    _rust_parallel_sort = None

# Import CPU and memory monitoring utilities
from utils import timing_wrapper_with_monitoring

# =============================================================================
# ALGORITHM CONSTANTS
# =============================================================================

# Algorithm name constants (no spaces, easy to use)
BUBBLE_SORT = "BUBBLE_SORT"
BUILT_IN_SORT = "BUILT_IN_SORT"
QUICK_SORT = "QUICK_SORT"
MERGE_SORT = "MERGE_SORT"
HEAP_SORT = "HEAP_SORT"
MLX_SORT = "mlx_sort"
MLX_SORT_PRELOAD_TO_MEMORY = "mlx_sort_preload_to_memory"
POLAR_SORT = "POLAR_SORT"
RUST_PARALLEL_SORT = "RUST_PARALLEL_SORT"

# Display names for user-friendly output
ALGORITHM_DISPLAY_NAMES = {
    BUBBLE_SORT: "Bubble Sort",
    BUILT_IN_SORT: "Built-in Sort",
    QUICK_SORT: "Quick Sort",
    MERGE_SORT: "Merge Sort",
    HEAP_SORT: "Heap Sort",
    MLX_SORT: "MLX Sort (incl. load)",
    MLX_SORT_PRELOAD_TO_MEMORY: "MLX Sort (preloaded)",
    POLAR_SORT: "Polar Sort",
    RUST_PARALLEL_SORT: "Rust Parallel Sort (Rayon)"
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def timing_wrapper(sort_func, arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
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


# =============================================================================
# BUBBLE SORT
# =============================================================================

def bubble_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of bubble sort algorithm.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    
    Args:
        arr: Input array to sort
        
    Returns:
        Sorted array
    """
    n = len(arr)
    sorted_arr = arr.copy()
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_arr[j] > sorted_arr[j + 1]:
                sorted_arr[j], sorted_arr[j + 1] = sorted_arr[j + 1], sorted_arr[j]
    
    return sorted_arr


def bubble_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Bubble sort with timing measurement.
    
    Args:
        arr: Input array to sort
        
    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(bubble_sort_impl, arr)


# =============================================================================
# BUILT-IN SORT
# =============================================================================

def built_in_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation using Python's built-in sorted() function.
    
    Time Complexity: O(n log n) - Timsort
    Space Complexity: O(n)
    
    Args:
        arr: Input array to sort
        
    Returns:
        Sorted array
    """
    return sorted(arr)


def built_in_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Built-in sort with timing measurement.
    
    Args:
        arr: Input array to sort
        
    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(built_in_sort_impl, arr)


# =============================================================================
# QUICK SORT
# =============================================================================

def quick_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of quick sort algorithm.
    
    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) average, O(n) worst case
    
    Args:
        arr: Input array to sort
        
    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort_impl(left) + middle + quick_sort_impl(right)


def quick_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Quick sort with timing measurement.
    
    Args:
        arr: Input array to sort
        
    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(quick_sort_impl, arr)


# =============================================================================
# MERGE SORT
# =============================================================================

def merge(left: List[Any], right: List[Any]) -> List[Any]:
    """
    Merge two sorted arrays into a single sorted array.
    
    Args:
        left: Left sorted array
        right: Right sorted array
        
    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of merge sort algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Args:
        arr: Input array to sort
        
    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort_impl(arr[:mid])
    right = merge_sort_impl(arr[mid:])
    
    return merge(left, right)


def merge_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Merge sort with timing measurement.
    
    Args:
        arr: Input array to sort
        
    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(merge_sort_impl, arr)


# =============================================================================
# HEAP SORT
# =============================================================================

def heapify(arr: List[Any], n: int, i: int) -> None:
    """
    Heapify subtree rooted at index i.
    
    Args:
        arr: Array to heapify
        n: Size of heap
        i: Index of subtree root
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort_impl(arr: List[Any]) -> List[Any]:
    """
    Implementation of heap sort algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Args:
        arr: Input array to sort
        
    Returns:
        Sorted array
    """
    n = len(arr)
    arr_copy = arr.copy()
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr_copy, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr_copy[0], arr_copy[i] = arr_copy[i], arr_copy[0]
        heapify(arr_copy, i, 0)
    
    return arr_copy


def heap_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Heap sort with timing measurement.
    
    Args:
        arr: Input array to sort
        
    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    return timing_wrapper(heap_sort_impl, arr)


# =============================================================================
# MLX/MPS TORCH SORT
# =============================================================================

def _get_mlx_torch_device() -> torch.device:
    """
    Resolve the MLX/MPS device on Apple Silicon; fallback to CPU if unavailable.
    """
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def _mps_synchronize_if_needed(device: torch.device) -> None:
    """Synchronize MPS device for accurate timing when using asynchronous ops."""
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            # Best-effort; if synchronize fails, continue
            pass


def mlx_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Torch-based sort on MLX/MPS including CPU->GPU transfer time.

    - Measures time for: host->device transfer + sort + device->host transfer.
    """
    device = _get_mlx_torch_device()
    with torch.no_grad():
        start_time = time.time()
        cpu_tensor = torch.tensor(arr, dtype=torch.float32)
        tensor = cpu_tensor.to(device)  # include H2D transfer in timing
        _mps_synchronize_if_needed(device)
        sorted_tensor, _ = torch.sort(tensor)
        _mps_synchronize_if_needed(device)
        end_time = time.time()
        # Transfer result back to CPU outside timing window
        result_list = sorted_tensor.to("cpu").tolist()
    return result_list, end_time - start_time, {} # No CPU metrics for MLX sort


def mlx_sort_preload_to_memory(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Torch-based sort on MLX/MPS excluding CPU->GPU transfer time.

    - Preloads data to device prior to timing.
    - Measures time for: sort + device->host transfer only.
    """
    device = _get_mlx_torch_device()
    with torch.no_grad():
        # Preload to device, not counted in timing
        preload_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
        _mps_synchronize_if_needed(device)

        start_time = time.time()
        sorted_tensor, _ = torch.sort(preload_tensor)
        _mps_synchronize_if_needed(device)
        end_time = time.time()
        # Transfer result back to CPU outside timing window
        result_list = sorted_tensor.to("cpu").tolist()
    return result_list, end_time - start_time, {} # No CPU metrics for MLX sort

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
        raise RuntimeError("rust_parallel module is not built/available. See README for build instructions.")
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
    BUBBLE_SORT: bubble_sort,
    BUILT_IN_SORT: built_in_sort,
    QUICK_SORT: quick_sort,
    MERGE_SORT: merge_sort,
    HEAP_SORT: heap_sort,
    MLX_SORT: mlx_sort,
    MLX_SORT_PRELOAD_TO_MEMORY: mlx_sort_preload_to_memory,
    POLAR_SORT: polar_sort,
    RUST_PARALLEL_SORT: rust_parallel_sort
}


def get_available_algorithms() -> List[str]:
    """
    Get list of available sorting algorithm constants.
    
    Returns:
        List of algorithm constants
    """
    return list(SORTING_ALGORITHMS.keys())


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
    # First try direct lookup (for constants)
    if name in SORTING_ALGORITHMS:
        return SORTING_ALGORITHMS[name]
    
    # Then try reverse lookup (for display names)
    for constant, display_name in ALGORITHM_DISPLAY_NAMES.items():
        if display_name == name:
            return SORTING_ALGORITHMS[constant]
    
    # If not found, raise error with available options
    available_constants = list(SORTING_ALGORITHMS.keys())
    available_display_names = list(ALGORITHM_DISPLAY_NAMES.values())
    raise KeyError(f"Algorithm '{name}' not found. Available constants: {available_constants}, Available display names: {available_display_names}")


def get_display_name(constant: str) -> str:
    """
    Get display name for an algorithm constant.
    
    Args:
        constant: Algorithm constant
        
    Returns:
        Display name for the algorithm
    """
    return ALGORITHM_DISPLAY_NAMES.get(constant, constant)







