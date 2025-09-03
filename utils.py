# all utils functions

import random
import psutil
import os
import gc
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional
from queue import Queue, Empty

# Set up logging for debugging thread issues
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# function that will create an unsorted list of numbers.
# the size of this list will be provided by the caller of this function.


def create_unsorted_list(size, upper_bound_multiplier=10):
    # On a Macbook M2 with 16GB RAM, let's conservatively cap the list size to 50 million elements.
    # This should use less than 4GB of RAM for a list of ints, and is unlikely to take more than 30 minutes to generate.
    # On a Mac with 24GB RAM, to limit RAM usage to 10GB for this program:
    # Each Python int in a list typically uses at least 28 bytes (CPython 3.8+), but for random.randint() the ints are small and may be more memory-efficient.
    # For safety, assume ~28 bytes per int (worst case, 64-bit Python).
    # 10 GB = 10 * 1024**3 = 10,737,418,240 bytes
    # MAX_REASONABLE_SIZE = 10,737,418,240 // 28 ≈ 383,479,223
    MAX_REASONABLE_SIZE = 380_000_000
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    if size > MAX_REASONABLE_SIZE:
        raise ValueError(
            f"Requested list size {size} exceeds the safe limit of {MAX_REASONABLE_SIZE} elements."
        )
    return [random.randint(1, upper_bound_multiplier * size) for _ in range(size)]


def get_memory_usage():
    """Get current memory usage in MB with error handling"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        if memory_info:
            return memory_info.rss / 1024 / 1024
        return None
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting memory usage: {e}")
        return None


def safe_thread_join(thread: threading.Thread, timeout: float = 1.0, force_cleanup: bool = True) -> bool:
    """
    Safely join a thread with proper error handling and force cleanup.

    Args:
        thread: Thread to join
        timeout: Timeout in seconds
        force_cleanup: Whether to attempt force cleanup on timeout

    Returns:
        True if thread joined successfully, False otherwise
    """
    if not thread.is_alive():
        return True

    try:
        thread.join(timeout=timeout)
        if thread.is_alive():
            if force_cleanup:
                logger.warning(f"Thread {thread.name} did not terminate, attempting force cleanup")
                # For daemon threads, we can't force kill, but we can log and continue
                # The process will clean them up on exit
                return False
        return not thread.is_alive()
    except Exception as e:
        logger.warning(f"Error joining thread {thread.name}: {e}")
        return False


def track_memory_usage(
    duration: float, sample_interval: float = 0.01
) -> Dict[str, float]:
    """
    Track memory usage during algorithm execution.

    Args:
        duration: How long to monitor (seconds)
        sample_interval: How often to sample memory usage (seconds) - default 0.01s for more accuracy

    Returns:
        Dictionary with memory utilization metrics
    """
    memory_samples = []
    stop_monitoring = threading.Event()
    monitor_thread: Optional[threading.Thread] = None

    def memory_monitor():
        try:
            while not stop_monitoring.is_set():
                try:
                    memory_mb = get_memory_usage()
                    memory_samples.append(memory_mb)
                    time.sleep(sample_interval)
                except (OSError, psutil.Error) as e:
                    logger.warning(f"Memory monitoring error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in memory monitoring: {e}")
                    break
        except Exception as e:
            logger.error(f"Critical error in memory monitoring thread: {e}")

    try:
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()

        # Wait for specified duration
        time.sleep(duration)
    except Exception as e:
        logger.error(f"Error during memory monitoring: {e}")
    finally:
        # Ensure monitoring is stopped
        stop_monitoring.set()

        # Safely join the thread
        if monitor_thread:
            if not safe_thread_join(monitor_thread, timeout=2.0):
                logger.warning(
                    "Memory monitoring thread did not terminate within timeout"
                )

    if not memory_samples:
        return {
            "peak_memory_mb": 0.0,
            "avg_memory_mb": 0.0,
            "memory_increase_mb": 0.0,
            "sample_count": 0,
        }

    # Calculate metrics
    peak_memory = max(memory_samples)
    avg_memory = sum(memory_samples) / len(memory_samples)
    initial_memory = memory_samples[0] if memory_samples else 0
    memory_increase = peak_memory - initial_memory

    return {
        "peak_memory_mb": peak_memory,
        "avg_memory_mb": avg_memory,
        "memory_increase_mb": memory_increase,
        "sample_count": len(memory_samples),
    }


def cleanup_memory():
    """Perform memory cleanup and garbage collection"""
    try:
        # Force garbage collection
        gc.collect()

        # Give the system a moment to clean up
        # 0.1 seconds is usually sufficient for garbage collection to complete,
        # but on some systems or under heavy memory pressure, it may not always be enough.
        # If you observe memory not being released quickly enough, consider increasing this value.
        time.sleep(0.1)
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {e}")


def get_cpu_count():
    """Get the number of CPU cores available"""
    try:
        return psutil.cpu_count(logical=True) or 1
    except Exception as e:
        logger.warning(f"Failed to get CPU count: {e}")
        return 1


def monitor_cpu_usage(
    duration: float, sample_interval: float = 0.1
) -> Dict[str, float]:
    """
    Monitor CPU usage during algorithm execution.

    Args:
        duration: How long to monitor (seconds)
        sample_interval: How often to sample CPU usage (seconds)

    Returns:
        Dictionary with CPU utilization metrics
    """
    cpu_samples = []
    stop_monitoring = threading.Event()
    monitor_thread: Optional[threading.Thread] = None

    def cpu_monitor():
        try:
            while not stop_monitoring.is_set():
                try:
                    # Get per-CPU usage
                    cpu_percent = psutil.cpu_percent(
                        interval=sample_interval, percpu=True
                    )
                    cpu_samples.append(cpu_percent)
                except (OSError, psutil.Error) as e:
                    logger.warning(f"CPU monitoring error: {e}")
                    # If per-CPU fails, try overall CPU
                    try:
                        overall_cpu = psutil.cpu_percent(interval=sample_interval)
                        cpu_count = get_cpu_count()
                        cpu_samples.append([overall_cpu] * cpu_count)
                    except Exception as fallback_error:
                        logger.warning(
                            f"CPU monitoring fallback also failed: {fallback_error}"
                        )
                        break
                except Exception as e:
                    logger.error(f"Unexpected error in CPU monitoring: {e}")
                    break
        except Exception as e:
            logger.error(f"Critical error in CPU monitoring thread: {e}")

    try:
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
        monitor_thread.start()

        # Wait for specified duration
        time.sleep(duration)
    except Exception as e:
        logger.error(f"Error during CPU monitoring: {e}")
    finally:
        # Ensure monitoring is stopped
        stop_monitoring.set()

        # Safely join the thread
        if monitor_thread:
            if not safe_thread_join(monitor_thread, timeout=2.0):
                logger.warning("CPU monitoring thread did not terminate within timeout")

    if not cpu_samples:
        return {
            "avg_cpu_percent": 0.0,
            "max_cpu_percent": 0.0,
            "parallelization_efficiency": 0.0,
            "cpu_cores_utilized": 0.0,
            "cpu_count": get_cpu_count(),
            "sample_count": 0,
        }

    # Calculate metrics
    total_cpu_percent = sum(sum(sample) for sample in cpu_samples)
    avg_cpu_percent = total_cpu_percent / len(cpu_samples)
    max_cpu_percent = max(sum(sample) for sample in cpu_samples)

    # Calculate parallelization efficiency
    cpu_count = get_cpu_count()
    parallelization_efficiency = (avg_cpu_percent / (cpu_count * 100)) * 100

    # Estimate number of cores effectively utilized
    cpu_cores_utilized = avg_cpu_percent / 100

    return {
        "avg_cpu_percent": avg_cpu_percent,
        "max_cpu_percent": max_cpu_percent,
        "parallelization_efficiency": parallelization_efficiency,
        "cpu_cores_utilized": cpu_cores_utilized,
        "cpu_count": cpu_count,
        "sample_count": len(cpu_samples),
    }


def timing_wrapper_with_cpu_monitoring(
    sort_func, arr: List, monitor_duration: float = None
) -> Tuple[List, float, Dict[str, float]]:
    """
    Enhanced timing wrapper that also monitors CPU usage during execution.

    Args:
        sort_func: The sorting function to wrap
        arr: Input array to sort
        monitor_duration: How long to monitor CPU (if None, monitors for full execution)

    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, cpu_metrics)
    """
    start_time = time.time()

    # Start CPU monitoring
    cpu_metrics = monitor_cpu_usage(0.1)  # Start with a short sample

    # Run the sorting algorithm
    sorted_arr = sort_func(arr.copy())

    end_time = time.time()
    execution_time = end_time - start_time

    # If we have a specific monitoring duration, use it
    if monitor_duration and execution_time > monitor_duration:
        # Re-run with proper monitoring duration
        cpu_metrics = monitor_cpu_usage(monitor_duration)
    else:
        # Monitor for the full execution time
        cpu_metrics = monitor_cpu_usage(execution_time)

    return sorted_arr, execution_time, cpu_metrics


def timing_wrapper_with_monitoring(
    sort_func, arr: List, monitor_duration: float = None
) -> Tuple[List, float, Dict[str, float]]:
    """
    Enhanced timing wrapper that monitors both CPU and memory usage during execution.
    Uses thread-safe queues to avoid race conditions.

    Args:
        sort_func: The sorting function to wrap
        arr: Input array to sort
        monitor_duration: How long to monitor (if None, monitors for full execution)

    Returns:
        Tuple of (sorted_array, execution_time_in_seconds, metrics_dict)
    """
    # Input validation
    if arr is None:
        raise ValueError("Input array cannot be None")
    if not isinstance(arr, (list, tuple)):
        raise TypeError("Input must be a list or tuple")

    # Use thread-safe queues instead of shared lists
    memory_queue: Queue = Queue()
    cpu_queue: Queue = Queue()
    stop_monitoring = threading.Event()
    memory_thread: Optional[threading.Thread] = None
    cpu_thread: Optional[threading.Thread] = None

    def memory_monitor():
        thread_name = "MemoryMonitor"
        try:
            while not stop_monitoring.is_set():
                try:
                    memory_mb = get_memory_usage()
                    if memory_mb is not None:
                        memory_queue.put(memory_mb)
                    
                    # Use wait with timeout instead of sleep for responsive shutdown
                    if stop_monitoring.wait(timeout=0.01):
                        break
                except (OSError, psutil.Error) as e:
                    logger.warning(f"{thread_name}: Memory monitoring error: {e}")
                    # Don't break immediately, try a few more times
                    if stop_monitoring.wait(timeout=0.1):
                        break
                except Exception as e:
                    logger.error(f"{thread_name}: Unexpected error: {e}")
                    break
        except Exception as e:
            logger.error(f"{thread_name}: Critical error: {e}")
        finally:
            # Always signal completion, even on error
            try:
                memory_queue.put(None)
                logger.debug(f"{thread_name}: Thread terminated cleanly")
            except Exception as e:
                logger.error(f"{thread_name}: Error during cleanup: {e}")

    def cpu_monitor():
        thread_name = "CPUMonitor"
        try:
            while not stop_monitoring.is_set():
                try:
                    # Use non-blocking CPU monitoring to avoid hanging
                    if stop_monitoring.is_set():
                        break
                        
                    # Get per-CPU usage with shorter interval for responsiveness
                    cpu_percent = psutil.cpu_percent(interval=0.005, percpu=True)
                    if cpu_percent:
                        cpu_queue.put(cpu_percent)
                        
                    # Check for stop signal more frequently
                    if stop_monitoring.wait(timeout=0.005):
                        break
                        
                except (OSError, psutil.Error) as e:
                    logger.warning(f"{thread_name}: CPU monitoring error: {e}")
                    # If per-CPU fails, try overall CPU with timeout check
                    if not stop_monitoring.is_set():
                        try:
                            overall_cpu = psutil.cpu_percent(interval=0.005)
                            cpu_count = get_cpu_count()
                            if not stop_monitoring.is_set():
                                cpu_queue.put([overall_cpu] * cpu_count)
                        except Exception as fallback_error:
                            logger.warning(
                                f"{thread_name}: Fallback failed: {fallback_error}"
                            )
                    # Brief pause before retry
                    if stop_monitoring.wait(timeout=0.1):
                        break
                except Exception as e:
                    logger.error(f"{thread_name}: Unexpected error: {e}")
                    break
        except Exception as e:
            logger.error(f"{thread_name}: Critical error: {e}")
        finally:
            # Always signal completion, even on error
            try:
                cpu_queue.put(None)
                logger.debug(f"{thread_name}: Thread terminated cleanly")
            except Exception as e:
                logger.error(f"{thread_name}: Error during cleanup: {e}")

    try:
        # Start monitoring in background threads with proper names
        memory_thread = threading.Thread(
            target=memory_monitor, 
            name="MemoryMonitor", 
            daemon=True
        )
        cpu_thread = threading.Thread(
            target=cpu_monitor, 
            name="CPUMonitor", 
            daemon=True
        )
        memory_thread.start()
        cpu_thread.start()
        
        # Verify threads started successfully
        if not memory_thread.is_alive():
            logger.warning("Memory monitoring thread failed to start")
        if not cpu_thread.is_alive():
            logger.warning("CPU monitoring thread failed to start")

        # Small delay to ensure monitoring starts
        time.sleep(0.01)

        # Run the sorting algorithm
        start_time = time.time()
        sorted_arr = sort_func(arr.copy())
        end_time = time.time()
        execution_time = end_time - start_time

    except Exception as e:
        logger.error(f"Error during algorithm execution: {e}")
        execution_time = 0.0
        sorted_arr = arr.copy()  # Return unsorted array as fallback
    finally:
        # Stop monitoring with escalating timeouts
        stop_monitoring.set()
        
        # First attempt: gentle shutdown with normal timeout
        memory_joined = False
        cpu_joined = False
        
        if memory_thread and memory_thread.is_alive():
            memory_joined = safe_thread_join(memory_thread, timeout=2.0)
            if not memory_joined:
                logger.warning(
                    f"Memory monitoring thread ({memory_thread.name}) did not terminate within 2s timeout"
                )

        if cpu_thread and cpu_thread.is_alive():
            cpu_joined = safe_thread_join(cpu_thread, timeout=2.0)
            if not cpu_joined:
                logger.warning(
                    f"CPU monitoring thread ({cpu_thread.name}) did not terminate within 2s timeout"
                )
        
        # Second attempt: extended timeout for stuck threads
        if not memory_joined and memory_thread and memory_thread.is_alive():
            logger.info("Attempting extended shutdown for memory thread...")
            if not safe_thread_join(memory_thread, timeout=5.0):
                logger.error(
                    f"Memory monitoring thread ({memory_thread.name}) is stuck and may leak resources"
                )
                
        if not cpu_joined and cpu_thread and cpu_thread.is_alive():
            logger.info("Attempting extended shutdown for CPU thread...")
            if not safe_thread_join(cpu_thread, timeout=5.0):
                logger.error(
                    f"CPU monitoring thread ({cpu_thread.name}) is stuck and may leak resources"
                )
        
        # Cleanup: Clear any remaining queue items to prevent memory leaks
        try:
            while not memory_queue.empty():
                memory_queue.get_nowait()
        except Empty:
            pass
        except Exception as e:
            logger.warning(f"Error cleaning memory queue: {e}")
            
        try:
            while not cpu_queue.empty():
                cpu_queue.get_nowait()
        except Empty:
            pass
        except Exception as e:
            logger.warning(f"Error cleaning CPU queue: {e}")

    # Collect all samples from thread-safe queues
    memory_samples = []
    cpu_samples = []
    
    # Drain memory queue with timeout protection
    queue_timeout = time.time() + 5.0  # 5 second timeout for queue draining
    try:
        while time.time() < queue_timeout:
            try:
                sample = memory_queue.get_nowait()
                if sample is None:  # End marker
                    break
                memory_samples.append(sample)
            except Empty:
                break
    except Exception as e:
        logger.warning(f"Error draining memory queue: {e}")
    
    # Drain CPU queue with timeout protection
    try:
        while time.time() < queue_timeout:
            try:
                sample = cpu_queue.get_nowait()
                if sample is None:  # End marker
                    break
                cpu_samples.append(sample)
            except Empty:
                break
    except Exception as e:
        logger.warning(f"Error draining CPU queue: {e}")
    
    # Log collection statistics
    logger.debug(f"Collected {len(memory_samples)} memory samples, {len(cpu_samples)} CPU samples")

    # Calculate memory metrics (now thread-safe)
    if memory_samples:
        peak_memory = max(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        initial_memory = memory_samples[0] if memory_samples else 0
        memory_increase = peak_memory - initial_memory
    else:
        peak_memory = avg_memory = memory_increase = 0.0

    # Calculate CPU metrics from samples collected during execution (now thread-safe)
    if cpu_samples:
        total_cpu_percent = sum(sum(sample) for sample in cpu_samples)
        avg_cpu_percent = total_cpu_percent / len(cpu_samples)
        max_cpu_percent = max(sum(sample) for sample in cpu_samples)

        # Calculate parallelization efficiency
        cpu_count = get_cpu_count()
        parallelization_efficiency = (avg_cpu_percent / (cpu_count * 100)) * 100

        # Estimate number of cores effectively utilized
        cpu_cores_utilized = avg_cpu_percent / 100
    else:
        avg_cpu_percent = max_cpu_percent = parallelization_efficiency = (
            cpu_cores_utilized
        ) = 0.0
        cpu_count = get_cpu_count()

    # Combine metrics
    combined_metrics = {
        "avg_cpu_percent": avg_cpu_percent,
        "max_cpu_percent": max_cpu_percent,
        "parallelization_efficiency": parallelization_efficiency,
        "cpu_cores_utilized": cpu_cores_utilized,
        "cpu_count": cpu_count,
        "peak_memory_mb": peak_memory,
        "avg_memory_mb": avg_memory,
        "memory_increase_mb": memory_increase,
        "memory_sample_count": len(memory_samples),
        "cpu_sample_count": len(cpu_samples),
    }

    return sorted_arr, execution_time, combined_metrics


# =============================================================================
# ALGORITHM REGISTRY UTILITIES
# =============================================================================


def get_available_algorithms(
    sorting_algorithms: Dict, algorithm_display_names: Dict
) -> List[str]:
    """
    Get list of available sorting algorithm constants.

    Args:
        sorting_algorithms: Dictionary mapping algorithm names to their wrapper functions
        algorithm_display_names: Dictionary mapping algorithm constants to display names

    Returns:
        List of algorithm constants
    """
    return list(sorting_algorithms.keys())


def get_algorithm(name: str, sorting_algorithms: Dict, algorithm_display_names: Dict):
    """
    Get sorting algorithm function by name.

    Args:
        name: Name of the algorithm (constant or display name)
        sorting_algorithms: Dictionary mapping algorithm names to their wrapper functions
        algorithm_display_names: Dictionary mapping algorithm constants to display names

    Returns:
        Algorithm wrapper function

    Raises:
        KeyError: If algorithm name is not found
    """
    # First try direct lookup (for constants)
    if name in sorting_algorithms:
        return sorting_algorithms[name]

    # Then try reverse lookup (for display names)
    for constant, display_name in algorithm_display_names.items():
        if display_name == name:
            return sorting_algorithms[constant]

    # If not found, raise error with available options
    available_constants = list(sorting_algorithms.keys())
    available_display_names = list(algorithm_display_names.values())
    raise KeyError(
        f"Algorithm '{name}' not found. Available constants: {available_constants}, Available display names: {available_display_names}"
    )


def get_display_name(constant: str, algorithm_display_names: Dict) -> str:
    """
    Get display name for an algorithm constant.

    Args:
        constant: Algorithm constant
        algorithm_display_names: Dictionary mapping algorithm constants to display names

    Returns:
        Display name for the algorithm
    """
    return algorithm_display_names.get(constant, constant)


def get_available_mlx_algorithms(mlx_sorting_algorithms: Dict) -> List[str]:
    """
    Get list of available MLX sorting algorithm constants.

    Args:
        mlx_sorting_algorithms: Dictionary mapping MLX algorithm names to their wrapper functions

    Returns:
        List of algorithm constants
    """
    return list(mlx_sorting_algorithms.keys())


def get_mlx_algorithm(
    name: str, mlx_sorting_algorithms: Dict, algorithm_display_names: Dict
):
    """
    Get MLX sorting algorithm function by name.

    Args:
        name: Name of the algorithm (constant or display name)
        mlx_sorting_algorithms: Dictionary mapping MLX algorithm names to their wrapper functions
        algorithm_display_names: Dictionary mapping algorithm constants to display names

    Returns:
        Algorithm wrapper function

    Raises:
        KeyError: If algorithm name is not found
    """
    # First try direct lookup (for constants)
    if name in mlx_sorting_algorithms:
        return mlx_sorting_algorithms[name]

    # Then try reverse lookup (for display names)
    for constant, display_name in algorithm_display_names.items():
        if display_name == name:
            return mlx_sorting_algorithms[constant]

    # If not found, raise error with available options
    available_constants = list(mlx_sorting_algorithms.keys())
    available_display_names = list(algorithm_display_names.values())
    raise KeyError(
        f"MLX Algorithm '{name}' not found. Available constants: {available_constants}, Available display names: {available_display_names}"
    )


def get_mlx_display_name(constant: str, algorithm_display_names: Dict) -> str:
    """
    Get display name for an MLX algorithm constant.

    Args:
        constant: Algorithm constant
        algorithm_display_names: Dictionary mapping algorithm constants to display names

    Returns:
        Display name for the algorithm
    """
    return algorithm_display_names.get(constant, constant)
