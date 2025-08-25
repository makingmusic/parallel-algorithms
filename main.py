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
    POLAR_SORT,
    RUST_PARALLEL_SORT,
)
from utils import create_unsorted_list, get_memory_usage, cleanup_memory, get_cpu_count

CONFIG = {
    "list_size": 10_000_000,
    # List of algorithm constants to test. If None, all algorithms will be tested.
    # Available constants: BUILT_IN_SORT, QUICK_SORT, BUBBLE_SORT, MERGE_SORT, HEAP_SORT, mlx_sort, mlx_sort_preload_to_memory, POLAR_SORT
    "algorithms_to_test": [
        BUILT_IN_SORT,
        MLX_SORT,
        MLX_SORT_PRELOAD_TO_MEMORY,
        POLAR_SORT,
        RUST_PARALLEL_SORT,
    ],  # Set to None to test all
}


def benchmark_sorting_algorithms():
    """Benchmark different sorting algorithms"""
    print(
        f"Creating unsorted list of {CONFIG['list_size']:,} numbers...",
        end="",
        flush=True,
    )
    unsorted_list = create_unsorted_list(CONFIG["list_size"])
    print("Done")

    # Get all available algorithms
    all_algorithms = SORTING_ALGORITHMS.items()

    # Filter algorithms based on configuration
    if CONFIG["algorithms_to_test"] is not None:
        algorithms = [
            (name, func)
            for name, func in all_algorithms
            if name in CONFIG["algorithms_to_test"]
        ]
        if not algorithms:
            raise ValueError(
                f"No valid algorithms found in {CONFIG['algorithms_to_test']}. Available: {list(SORTING_ALGORITHMS.keys())}"
            )
        print(
            f"Testing selected algorithms: {', '.join([get_display_name(name) for name, _ in algorithms])}"
        )
    else:
        algorithms = all_algorithms
        print(
            f"Testing all available algorithms: {', '.join([get_display_name(name) for name, _ in algorithms])}"
        )

    results = []

    print("\nBenchmarking sorting algorithms...")
    print("=" * 80)

    for name, algorithm in algorithms:
        display_name = get_display_name(name)
        print(f"Testing {display_name}...", end="", flush=True)

        # Clean up memory before each benchmark
        cleanup_memory()
        
        # Run the sorting algorithm (includes memory tracking)
        sorted_list, execution_time, metrics = algorithm(unsorted_list)
        
        # Extract memory and CPU metrics
        memory_increase = metrics.get('memory_increase_mb', 0.0)
        peak_memory = metrics.get('peak_memory_mb', 0.0)
        
        # Use memory increase if positive, otherwise show peak memory usage
        memory_used = memory_increase if memory_increase > 0 else peak_memory
        
        # Format memory display to show both peak and increase when relevant
        if memory_increase > 0 and memory_increase < peak_memory * 0.9:
            memory_display = f"{peak_memory:.1f}/{memory_increase:.1f}"
        else:
            memory_display = f"{memory_used:.1f}"
        
        cpu_metrics = {k: v for k, v in metrics.items() if k not in ['peak_memory_mb', 'avg_memory_mb', 'memory_increase_mb', 'sample_count']}
        
        # Verify the list is sorted
        is_sorted = all(
            sorted_list[i] <= sorted_list[i + 1] for i in range(len(sorted_list) - 1)
        )

        results.append(
            {
                "algorithm": display_name,
                "time": execution_time,
                "memory": memory_used,
            'memory_increase': memory_increase,
            'peak_memory': peak_memory,
            'memory_display': memory_display,
                "sorted": is_sorted,
                "cpu_metrics": cpu_metrics,
            }
        )

        print(f"  ✓ completed in {execution_time:.4f} seconds")

    return results


def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "=" * 120)
    print("SORTING ALGORITHM BENCHMARK RESULTS")
    print("=" * 120)

    # Calculate column widths based on content
    max_algorithm_len = max(len(result["algorithm"]) for result in results)
    algorithm_width = max(25, max_algorithm_len + 2)

    # Table header with proper spacing
    print(f"{'Algorithm':<{algorithm_width}} {'Time (s)':<12} {'Memory (MB)':<15} {'CPU Eff. (%)':<12} {'Cores Used':<12} {'Status':<12}")
    print(f"{'':<{algorithm_width}} {'':<12} {'(Peak/Increase)':<15} {'':<12} {'':<12} {'':<12}")
    print(
        f"{'Algorithm':<{algorithm_width}} {'Time (s)':<12} {'Memory (MB)':<15} {'CPU Eff. (%)':<12} {'Cores Used':<12} {'Status':<12}"
    )
    print("-" * 120)

    # Sort results by execution time
    sorted_results = sorted(results, key=lambda x: x["time"])

    for result in sorted_results:
        status = "✓ Sorted" if result["sorted"] else "✗ Failed"

        # Extract CPU metrics
        cpu_metrics = result.get("cpu_metrics", {})
        cpu_efficiency = cpu_metrics.get("parallelization_efficiency", 0.0)
        cores_used = cpu_metrics.get("cpu_cores_utilized", 0.0)

        print(
            f"{result['algorithm']:<{algorithm_width}} {result['time']:<12.4f} {result['memory_display']:<15} {cpu_efficiency:<12.1f} {cores_used:<12.1f} {status:<12}"
        )

    print("-" * 120)

    # Summary
    fastest = min(results, key=lambda x: x["time"])
    slowest = max(results, key=lambda x: x["time"])

    print(f"\nFastest: {fastest['algorithm']} ({fastest['time']:.4f}s)")
    print(f"Slowest: {slowest['algorithm']} ({slowest['time']:.4f}s)")

    # Performance comparison
    if fastest["algorithm"] != slowest["algorithm"]:
        speedup = slowest["time"] / fastest["time"]
        print(
            f"Speedup: {fastest['algorithm']} is {speedup:.1f}x faster than {slowest['algorithm']}"
        )

    # Parallelization summary
    print(f"\nParallelization Analysis:")
    print(f"Available CPU cores: {get_cpu_count()}")

    # Find best parallelized algorithm
    parallel_results = [
        r
        for r in results
        if r.get("cpu_metrics", {}).get("parallelization_efficiency", 0) > 0
    ]
    if parallel_results:
        best_parallel = max(
            parallel_results,
            key=lambda x: x["cpu_metrics"]["parallelization_efficiency"],
        )
        print(
            f"Best parallelization: {best_parallel['algorithm']} ({best_parallel['cpu_metrics']['parallelization_efficiency']:.1f}% efficiency)"
        )

    # Show efficiency ranges
    efficiencies = [
        r.get("cpu_metrics", {}).get("parallelization_efficiency", 0) for r in results
    ]
    if efficiencies:
        print(f"Efficiency range: {min(efficiencies):.1f}% - {max(efficiencies):.1f}%")

    # Detailed parallelization breakdown
    print(f"\nDetailed Parallelization Breakdown:")
    print("-" * 80)
    for result in results:
        cpu_metrics = result.get("cpu_metrics", {})
        efficiency = cpu_metrics.get("parallelization_efficiency", 0.0)
        cores_used = cpu_metrics.get("cpu_cores_utilized", 0.0)
        avg_cpu = cpu_metrics.get("avg_cpu_percent", 0.0)
        max_cpu = cpu_metrics.get("max_cpu_percent", 0.0)

        if efficiency > 0:
            print(
                f"{result['algorithm']:<30} {efficiency:>6.1f}% efficiency, {cores_used:>4.1f} cores, {avg_cpu:>6.1f}% avg CPU"
            )
        else:
            print(
                f"{result['algorithm']:<30} {'GPU/Other':>6} (CPU metrics not applicable)"
            )

    # Performance vs Parallelization analysis
    print(f"\nPerformance vs Parallelization Analysis:")
    print("-" * 80)

    # Find algorithms that are both fast and well-parallelized
    fast_threshold = min(r["time"] for r in results) * 2  # Within 2x of fastest
    efficient_threshold = 10.0  # At least 10% efficiency

    fast_and_efficient = [
        r
        for r in results
        if r["time"] <= fast_threshold
        and r.get("cpu_metrics", {}).get("parallelization_efficiency", 0)
        >= efficient_threshold
    ]

    if fast_and_efficient:
        print(
            f"Fast AND well-parallelized algorithms (≤{fast_threshold:.3f}s, ≥{efficient_threshold}% efficiency):"
        )
        for result in fast_and_efficient:
            efficiency = result["cpu_metrics"]["parallelization_efficiency"]
            print(
                f"  • {result['algorithm']}: {result['time']:.3f}s, {efficiency:.1f}% efficiency"
            )
    else:
        print(
            f"No algorithms are both fast (≤{fast_threshold:.3f}s) and well-parallelized (≥{efficient_threshold}% efficiency)"
        )

    # Recommendations
    print(f"\nRecommendations:")
    fastest = min(results, key=lambda x: x["time"])
    most_efficient = (
        max(
            parallel_results,
            key=lambda x: x["cpu_metrics"]["parallelization_efficiency"],
        )
        if parallel_results
        else None
    )

    print(f"  • For pure speed: {fastest['algorithm']} ({fastest['time']:.3f}s)")
    if most_efficient:
        print(
            f"  • For CPU utilization: {most_efficient['algorithm']} ({most_efficient['cpu_metrics']['parallelization_efficiency']:.1f}% efficiency)"
        )

    # GPU vs CPU analysis
    gpu_algorithms = [r for r in results if "MLX" in r["algorithm"]]
    cpu_algorithms = [r for r in results if "MLX" not in r["algorithm"]]

    if gpu_algorithms and cpu_algorithms:
        fastest_gpu = min(gpu_algorithms, key=lambda x: x["time"])
        fastest_cpu = min(cpu_algorithms, key=lambda x: x["time"])
        print(
            f"  • GPU vs CPU: {fastest_gpu['algorithm']} ({fastest_gpu['time']:.3f}s) vs {fastest_cpu['algorithm']} ({fastest_cpu['time']:.3f}s)"
        )


def main():
    """Main function to run the benchmarking"""
    try:
        print("Parallel Algorithms - Sorting Benchmark")
        print("=" * 50)

        # Display available algorithms
        # print(f"Available algorithms: {', '.join(get_available_algorithms())}")

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
