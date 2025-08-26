#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import time
import random
import logging
from sort import SORTING_ALGORITHMS, BUILT_IN_SORT, POLAR_SORT, MLX_SORT

# Set up logging for debugging
logger = logging.getLogger(__name__)

def test_setup():
    print("🧪 Testing setup...")
    
    # Create a small test list
    test_list = [random.randint(1, 1000) for _ in range(1000)]
    
    # Test a few algorithms
    algorithms = [
        ("Built-in Sort", SORTING_ALGORITHMS[BUILT_IN_SORT]),
        ("Polar Sort", SORTING_ALGORITHMS[POLAR_SORT]),
        ("MLX Sort", SORTING_ALGORITHMS[MLX_SORT]),
    ]
    
    for name, algorithm in algorithms:
        try:
            start_time = time.time()
            sorted_list, execution_time, cpu_metrics = algorithm(test_list)
            end_time = time.time()
            
            # Verify sorting
            is_sorted = all(sorted_list[i] <= sorted_list[i + 1] for i in range(len(sorted_list) - 1))
            
            if is_sorted:
                print(f"✅ {name}: {execution_time:.4f}s")
            else:
                print(f"❌ {name}: Failed to sort correctly")
                
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Algorithm {name} failed with expected error: {e}")
            print(f"❌ {name}: Error - {e}")
        except Exception as e:
            logger.error(f"Algorithm {name} failed with unexpected error: {e}")
            print(f"❌ {name}: Unexpected error - {e}")
    
    print("🎉 Setup test completed!")

if __name__ == "__main__":
    test_setup()
