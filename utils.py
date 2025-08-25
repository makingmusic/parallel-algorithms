# all utils functions

import random
import psutil
import os
import gc
import time

# function that will create an unsorted list of numbers. 
# the size of this list will be provided by the caller of this function. 

def create_unsorted_list(size, upper_bound_multiplier=10):
    # On a Macbook M2 with 16GB RAM, let's conservatively cap the list size to 50 million elements.
    # This should use less than 4GB of RAM for a list of ints, and is unlikely to take more than 30 minutes to generate.
    MAX_REASONABLE_SIZE = 500_000_000
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

