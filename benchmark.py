### Runs runtime benchmark for certain functions for certain number of samples

import time
import numpy as np



def benchmark_functions (funcs, *args, num_samples=10, **kwargs):
    times = {f.__name__: [] for f in funcs}
    for _ in range(num_samples):
        results = {f.__name__: [] for f in funcs}
        for f in funcs:
            start_time = time.time()
            res = f(*args, **kwargs)
            end_time = time.time()
            
            times[f.__name__].append(end_time - start_time)
            results[f.__name__].append(res)

        assert all(np.array_equal(results[funcs[0].__name__], results[f.__name__]) for f in funcs), "Function results do not match!"

    return times