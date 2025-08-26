#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import time
import random
from sort import built_in_sort, polar_sort, mlx_sort


def test_setup():
    print("🧪 Testing setup...")

    # Create a small test list
    test_list = [random.randint(1, 1000) for _ in range(1000)]

    # Test a few algorithms
    algorithms = [
        ("Built-in Sort", built_in_sort),
        ("Polar Sort", polar_sort),
        ("MLX Sort", mlx_sort),
    ]

    for name, algorithm in algorithms:
        try:
            start_time = time.time()
            sorted_list, execution_time, cpu_metrics = algorithm(test_list)
            end_time = time.time()

            # Verify sorting
            is_sorted = all(
                sorted_list[i] <= sorted_list[i + 1]
                for i in range(len(sorted_list) - 1)
            )

            if is_sorted:
                print(f"✅ {name}: {execution_time:.4f}s")
            else:
                print(f"❌ {name}: Failed to sort correctly")

        except Exception as e:
            print(f"❌ {name}: Error - {e}")

    print("🎉 Setup test completed!")


if __name__ == "__main__":
    test_setup()
