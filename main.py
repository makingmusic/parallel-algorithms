import time
import random
import psutil
import os
from sort import (
    SORTING_ALGORITHMS,
    get_available_algorithms,
    get_display_name,
    BUILT_IN_SORT,
    QUICK_SORT,
    MLX_SORT,
    MLX_SORT_PRELOAD_TO_MEMORY,
    HEAP_SORT,
    BUBBLE_SORT,
    MERGE_SORT,
    POLAR_SORT
)
from utils import create_unsorted_list, get_memory_usage, cleanup_memory

CONFIG = {
    "list_size": 10_000_000,
    # List of algorithm constants to test. If None, all algorithms will be tested.
    # Available constants: BUILT_IN_SORT, QUICK_SORT, BUBBLE_SORT, MERGE_SORT, HEAP_SORT, mlx_sort, mlx_sort_preload_to_memory, POLAR_SORT
    "algorithms_to_test": [BUILT_IN_SORT, MLX_SORT, MLX_SORT_PRELOAD_TO_MEMORY, POLAR_SORT],  # Set to None to test all
}

def benchmark_sorting_algorithms():
    """Benchmark different sorting algorithms"""
    print(f"Creating unsorted list of {CONFIG['list_size']:,} numbers...", end="", flush=True)
    unsorted_list = create_unsorted_list(CONFIG['list_size'])
    print("Done")
    
    # Get all available algorithms
    all_algorithms = SORTING_ALGORITHMS.items()
    
    # Filter algorithms based on configuration
    if CONFIG['algorithms_to_test'] is not None:
        algorithms = [(name, func) for name, func in all_algorithms if name in CONFIG['algorithms_to_test']]
        if not algorithms:
            raise ValueError(f"No valid algorithms found in {CONFIG['algorithms_to_test']}. Available: {list(SORTING_ALGORITHMS.keys())}")
        print(f"Testing selected algorithms: {', '.join([get_display_name(name) for name, _ in algorithms])}")
    else:
        algorithms = all_algorithms
        print(f"Testing all available algorithms: {', '.join([get_display_name(name) for name, _ in algorithms])}")
    
    results = []
    
    print("\nBenchmarking sorting algorithms...")
    print("=" * 80)
    
    for name, algorithm in algorithms:
        display_name = get_display_name(name)
        print(f"Testing {display_name}...", end="", flush=True)
        
        # Clean up memory before each benchmark
        cleanup_memory()
        
        # Measure initial memory
        initial_memory = get_memory_usage()
        
        # Run the sorting algorithm
        sorted_list, execution_time = algorithm(unsorted_list)
        
        # Measure final memory
        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Verify the list is sorted
        is_sorted = all(sorted_list[i] <= sorted_list[i+1] for i in range(len(sorted_list)-1))
        
        results.append({
            'algorithm': display_name,
            'time': execution_time,
            'memory': memory_used,
            'sorted': is_sorted
        })
        
        print(f"  ✓ completed in {execution_time:.4f} seconds")
    
    return results

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "=" * 100)
    print("SORTING ALGORITHM BENCHMARK RESULTS")
    print("=" * 100)
    
    # Calculate column widths based on content
    max_algorithm_len = max(len(result['algorithm']) for result in results)
    algorithm_width = max(25, max_algorithm_len + 2)
    
    # Table header with proper spacing
    print(f"{'Algorithm':<{algorithm_width}} {'Time (s)':<12} {'Memory (MB)':<15} {'Status':<12}")
    print("-" * 100)
    
    # Sort results by execution time
    sorted_results = sorted(results, key=lambda x: x['time'])
    
    for result in sorted_results:
        status = "✓ Sorted" if result['sorted'] else "✗ Failed"
        print(f"{result['algorithm']:<{algorithm_width}} {result['time']:<12.4f} {result['memory']:<15.2f} {status:<12}")
    
    print("-" * 100)
    
    # Summary
    fastest = min(results, key=lambda x: x['time'])
    slowest = max(results, key=lambda x: x['time'])
    
    print(f"\nFastest: {fastest['algorithm']} ({fastest['time']:.4f}s)")
    print(f"Slowest: {slowest['algorithm']} ({slowest['time']:.4f}s)")
    
    # Performance comparison
    if fastest['algorithm'] != slowest['algorithm']:
        speedup = slowest['time'] / fastest['time']
        print(f"Speedup: {fastest['algorithm']} is {speedup:.1f}x faster than {slowest['algorithm']}")

def main():
    """Main function to run the benchmarking"""
    try:
        print("Parallel Algorithms - Sorting Benchmark")
        print("=" * 50)
        
        # Display available algorithms
        #print(f"Available algorithms: {', '.join(get_available_algorithms())}")
        
        # Run benchmarks
        results = benchmark_sorting_algorithms()
        
        # Display results
        print_results_table(results)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())



