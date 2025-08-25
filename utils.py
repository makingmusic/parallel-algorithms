# all utils functions

import random
import psutil
import os
import gc
import time
import threading
from typing import Dict, List, Tuple

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
    MAX_REASONABLE_SIZE = 380_000_000_0
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    if size > MAX_REASONABLE_SIZE:
        raise ValueError(f"Requested list size {size} exceeds the safe limit of {MAX_REASONABLE_SIZE} elements.")
    return [random.randint(1, upper_bound_multiplier*size) for _ in range(size)]

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Perform memory cleanup and garbage collection"""
    # Force garbage collection
    gc.collect()
    
    # Give the system a moment to clean up
    # 0.1 seconds is usually sufficient for garbage collection to complete,
    # but on some systems or under heavy memory pressure, it may not always be enough.
    # If you observe memory not being released quickly enough, consider increasing this value.
    time.sleep(0.1)

def get_cpu_count():
    """Get the number of CPU cores available"""
    return psutil.cpu_count(logical=True)

def monitor_cpu_usage(duration: float, sample_interval: float = 0.1) -> Dict[str, float]:
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
    
    def cpu_monitor():
        while not stop_monitoring.is_set():
            try:
                # Get per-CPU usage
                cpu_percent = psutil.cpu_percent(interval=sample_interval, percpu=True)
                cpu_samples.append(cpu_percent)
            except Exception:
                # If per-CPU fails, try overall CPU
                try:
                    overall_cpu = psutil.cpu_percent(interval=sample_interval)
                    cpu_samples.append([overall_cpu] * psutil.cpu_count())
                except Exception:
                    break
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=cpu_monitor, daemon=True)
    monitor_thread.start()
    
    # Wait for specified duration
    time.sleep(duration)
    stop_monitoring.set()
    monitor_thread.join(timeout=1.0)
    
    if not cpu_samples:
        return {
            'avg_cpu_percent': 0.0,
            'max_cpu_percent': 0.0,
            'parallelization_efficiency': 0.0,
            'cpu_cores_utilized': 0.0,
            'cpu_count': get_cpu_count()
        }
    
    # Calculate metrics
    total_cpu_percent = sum(sum(sample) for sample in cpu_samples)
    avg_cpu_percent = total_cpu_percent / len(cpu_samples)
    max_cpu_percent = max(sum(sample) for sample in cpu_samples)
    
    # Calculate parallelization efficiency
    # Efficiency = (average CPU usage across all cores) / (number of cores * 100%)
    cpu_count = get_cpu_count()
    parallelization_efficiency = (avg_cpu_percent / (cpu_count * 100)) * 100
    
    # Estimate number of cores effectively utilized
    # This is a rough estimate based on average CPU usage
    cpu_cores_utilized = avg_cpu_percent / 100
    
    return {
        'avg_cpu_percent': avg_cpu_percent,
        'max_cpu_percent': max_cpu_percent,
        'parallelization_efficiency': parallelization_efficiency,
        'cpu_cores_utilized': cpu_cores_utilized,
        'cpu_count': cpu_count,
        'sample_count': len(cpu_samples)
    }

def timing_wrapper_with_cpu_monitoring(sort_func, arr: List, monitor_duration: float = None) -> Tuple[List, float, Dict[str, float]]:
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

