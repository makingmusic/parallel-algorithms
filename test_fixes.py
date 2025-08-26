#!/usr/bin/env python3
"""
Test script to verify the fixes for:
1. CPU monitoring during algorithm execution
2. MLX sort precision preservation
"""

import time
import sys
import os
import logging

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sort import mlx_sort, mlx_sort_preload_to_memory, bubble_sort
from utils import timing_wrapper_with_monitoring

# Set up logging for debugging
logger = logging.getLogger(__name__)


def test_cpu_monitoring_fix():
    """Test that CPU monitoring happens during algorithm execution, not after."""
    print("Testing CPU monitoring fix...")
    
    # Create a test array that will take some time to sort
    test_array = list(range(100000, 0, -1))  # Reverse sorted array
    
    print("Running bubble sort with monitoring...")
    start_time = time.time()
    sorted_arr, exec_time, metrics = bubble_sort(test_array)
    end_time = time.time()
    
    print(f"Algorithm execution time: {exec_time:.4f} seconds")
    print(f"Total time including monitoring: {end_time - start_time:.4f} seconds")
    print(f"CPU metrics: {metrics}")
    
    # Verify that CPU monitoring happened during execution
    if metrics.get('cpu_sample_count', 0) > 0:
        print("✅ CPU monitoring fix: CPU samples collected during execution")
    else:
        print("❌ CPU monitoring fix: No CPU samples collected")
    
    # Verify that execution time is reasonable (not much longer than expected)
    if exec_time < 5.0:  # Bubble sort on 100k elements should take a few seconds
        print("✅ CPU monitoring fix: Execution time is reasonable")
    else:
        print("❌ CPU monitoring fix: Execution time seems too long")


def test_mlx_precision_fix():
    """Test that MLX sort preserves precision for large integers."""
    print("\nTesting MLX precision fix...")
    
    # Test with integers that would lose precision in float32
    large_integers = [
        16777216,  # 2^24 - this would be rounded in float32
        16777217,  # 2^24 + 1 - this would be rounded to 16777216 in float32
        16777218,  # 2^24 + 2 - this would be rounded to 16777218 in float32
        2147483647,  # Max int32
        2147483648,  # Max int32 + 1
        9223372036854775807,  # Max int64
    ]
    
    # Create a test array with these large integers in random order
    test_array = [large_integers[2], large_integers[0], large_integers[1], 
                  large_integers[4], large_integers[3], large_integers[5]]
    
    print(f"Original array: {test_array}")
    print(f"Expected sorted: {sorted(test_array)}")
    
    try:
        # Test both MLX sort functions
        print("\nTesting mlx_sort...")
        sorted_arr1, exec_time1, metrics1 = mlx_sort(test_array)
        print(f"MLX sort result: {sorted_arr1}")
        print(f"Execution time: {exec_time1:.4f} seconds")
        
        print("\nTesting mlx_sort_preload_to_memory...")
        sorted_arr2, exec_time2, metrics2 = mlx_sort_preload_to_memory(test_array)
        print(f"MLX sort (preloaded) result: {sorted_arr2}")
        print(f"Execution time: {exec_time2:.4f} seconds")
        
        # Verify that results are correct
        expected = sorted(test_array)
        
        if sorted_arr1 == expected:
            print("✅ MLX precision fix: mlx_sort preserves precision correctly")
        else:
            print("❌ MLX precision fix: mlx_sort lost precision")
            print(f"Expected: {expected}")
            print(f"Got: {sorted_arr1}")
        
        if sorted_arr2 == expected:
            print("✅ MLX precision fix: mlx_sort_preload_to_memory preserves precision correctly")
        else:
            print("❌ MLX precision fix: mlx_sort_preload_to_memory lost precision")
            print(f"Expected: {expected}")
            print(f"Got: {sorted_arr2}")
            
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"MLX sort failed with expected error: {e}")
        print(f"❌ MLX precision fix: Expected error during MLX sort: {e}")
    except Exception as e:
        logger.error(f"MLX sort failed with unexpected error: {e}")
        print(f"❌ MLX precision fix: Unexpected error during MLX sort: {e}")


def test_mixed_data_types():
    """Test MLX sort with mixed data types."""
    print("\nTesting MLX sort with mixed data types...")
    
    # Test with mixed integers and floats
    mixed_array = [1, 2.5, 3, 4.7, 5, 6.1, 7, 8.9]
    
    try:
        sorted_arr, exec_time, metrics = mlx_sort(mixed_array)
        expected = sorted(mixed_array)
        
        print(f"Original: {mixed_array}")
        print(f"Expected: {expected}")
        print(f"MLX result: {sorted_arr}")
        
        if sorted_arr == expected:
            print("✅ MLX precision fix: Mixed data types handled correctly")
        else:
            print("❌ MLX precision fix: Mixed data types not handled correctly")
            
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"MLX sort with mixed data types failed with expected error: {e}")
        print(f"❌ MLX precision fix: Expected error with mixed data types: {e}")
    except Exception as e:
        logger.error(f"MLX sort with mixed data types failed with unexpected error: {e}")
        print(f"❌ MLX precision fix: Unexpected error with mixed data types: {e}")


if __name__ == "__main__":
    print("Running tests for the fixes...")
    
    test_cpu_monitoring_fix()
    test_mlx_precision_fix()
    test_mixed_data_types()
    
    print("\nTest completed!")
