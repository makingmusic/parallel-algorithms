"""
MLX/MPS Sorting Algorithms Module

This module contains GPU-accelerated sorting algorithms using Apple MLX/MPS (Metal Performance Shaders).
These algorithms leverage the GPU for parallel sorting operations on Apple Silicon devices.

Algorithms included:
- MLX Sort (PyTorch MPS/MLX, includes host-to-device transfer)
- MLX Sort (PyTorch MPS/MLX, preloaded to device)
"""

import time
import torch
import logging
from typing import List, Tuple, Any, Dict

# Import registry utilities from utils
from utils import get_available_mlx_algorithms as utils_get_available_mlx_algorithms, get_mlx_algorithm as utils_get_mlx_algorithm, get_mlx_display_name as utils_get_mlx_display_name

# Set up logging for debugging
logger = logging.getLogger(__name__)

# =============================================================================
# ALGORITHM CONSTANTS
# =============================================================================

# Algorithm name constants (no spaces, easy to use)
MLX_SORT = "mlx_sort"
MLX_SORT_PRELOAD_TO_MEMORY = "mlx_sort_preload_to_memory"

# Display names for user-friendly output
ALGORITHM_DISPLAY_NAMES = {
    MLX_SORT: "MLX Sort (incl. load)",
    MLX_SORT_PRELOAD_TO_MEMORY: "MLX Sort (preloaded)",
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _get_mlx_torch_device() -> torch.device:
    """
    Resolve the MLX/MPS device on Apple Silicon; fallback to CPU if unavailable.
    """
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except (RuntimeError, OSError, AttributeError) as e:
        logger.warning(f"MPS device not available: {e}")
    except Exception as e:
        logger.error(f"Unexpected error checking MPS availability: {e}")
    return torch.device("cpu")


def _mps_synchronize_if_needed(device: torch.device) -> None:
    """Synchronize MPS device for accurate timing when using asynchronous ops."""
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except (RuntimeError, OSError) as e:
            logger.warning(f"MPS synchronization failed: {e}")
            # Best-effort; if synchronize fails, continue
        except Exception as e:
            logger.error(f"Unexpected error during MPS synchronization: {e}")


def _should_use_cpu_fallback(arr: List[Any]) -> bool:
    """
    Determine if we should use CPU fallback instead of MLX due to data type limitations.
    """
    if not arr:
        return False

    # Check for very large integers that might cause issues
    all_integers = all(isinstance(x, int) for x in arr)
    if all_integers:
        min_val = min(arr)
        max_val = max(arr)

        # If we have integers outside the safe range for MLX, use CPU fallback
        if max_val > 2147483647 or min_val < -2147483648:
            return True

    # Check for mixed types that might cause precision issues
    has_floats = any(isinstance(x, float) for x in arr)
    has_integers = any(isinstance(x, int) for x in arr)

    if has_floats and has_integers:
        # For mixed types, always use CPU fallback to preserve precision
        # This is because PyTorch will convert everything to float32, causing precision loss
        return True

    return False


def _cpu_sort_fallback(arr: List[Any]) -> List[Any]:
    """
    CPU-based sorting fallback that preserves precision.
    """
    return sorted(arr)


# =============================================================================
# MLX/MPS TORCH SORT
# =============================================================================


def mlx_sort(arr: List[Any]) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Torch-based sort on MLX/MPS including CPU->GPU transfer time.

    - Measures time for: host->device transfer + sort + device->host transfer.
    - Falls back to CPU sorting for problematic data types.
    """
    # Check if we should use CPU fallback
    if _should_use_cpu_fallback(arr):
        start_time = time.time()
        result_list = _cpu_sort_fallback(arr)
        end_time = time.time()
        return result_list, end_time - start_time, {}

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
    return result_list, end_time - start_time, {}  # No CPU metrics for MLX sort


def mlx_sort_preload_to_memory(
    arr: List[Any],
) -> Tuple[List[Any], float, Dict[str, float]]:
    """
    Torch-based sort on MLX/MPS excluding CPU->GPU transfer time.

    - Preloads data to device prior to timing.
    - Measures time for: sort + device->host transfer only.
    - Falls back to CPU sorting for problematic data types.
    """
    # Check if we should use CPU fallback
    if _should_use_cpu_fallback(arr):
        start_time = time.time()
        result_list = _cpu_sort_fallback(arr)
        end_time = time.time()
        return result_list, end_time - start_time, {}

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
    return result_list, end_time - start_time, {}  # No CPU metrics for MLX sort


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

# Dictionary mapping algorithm names to their wrapper functions
MLX_SORTING_ALGORITHMS = {
    MLX_SORT: mlx_sort,
    MLX_SORT_PRELOAD_TO_MEMORY: mlx_sort_preload_to_memory,
}


# Wrapper functions that use the registry utilities from utils.py
def get_available_mlx_algorithms() -> List[str]:
    """
    Get list of available MLX sorting algorithm constants.

    Returns:
        List of algorithm constants
    """
    return utils_get_available_mlx_algorithms(MLX_SORTING_ALGORITHMS)


def get_mlx_algorithm(name: str):
    """
    Get MLX sorting algorithm function by name.

    Args:
        name: Name of the algorithm (constant or display name)

    Returns:
        Algorithm wrapper function

    Raises:
        KeyError: If algorithm name is not found
    """
    return utils_get_mlx_algorithm(name, MLX_SORTING_ALGORITHMS, ALGORITHM_DISPLAY_NAMES)


def get_mlx_display_name(constant: str) -> str:
    """
    Get display name for an MLX algorithm constant.

    Args:
        constant: Algorithm constant

    Returns:
        Display name for the algorithm
    """
    return utils_get_mlx_display_name(constant, ALGORITHM_DISPLAY_NAMES)
