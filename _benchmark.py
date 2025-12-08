### Runs runtime benchmark for certain functions for certain number of samples

import time
import numpy as np


def benchmark_functions (funcs: list, *args: tuple, num_samples: int=10, **kwargs: dict) -> dict:
    """
    Benchmark multiple functions by running them num_samples times and recording their execution times. For the sake of compiled warp kernels, we run each function once that is not timed.

    Args:
        funcs: List of functions to benchmark.
        *args: Positional arguments to pass to each function.
        num_samples: Number of times to run each function for averaging.
        **kwargs: Keyword arguments to pass to each function.
    Returns:
        dict: A dictionary mapping function names to lists of execution times.
    """
    times = {f.__name__: [] for f in funcs}
    for i in range(num_samples + 1):
        results = {f.__name__: [] for f in funcs}
        for f in funcs:
            start_time = time.time()
            res = f(*args, **kwargs)
            end_time = time.time()

            if i > 0:
                times[f.__name__].append(end_time - start_time)
                results[f.__name__].append(res)

        # assert all(np.allclose(results[funcs[0].__name__], results[f.__name__], atol=1e-5) for f in funcs), "Function results do not match!"

    return times