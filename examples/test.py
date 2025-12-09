### Just testing warp functions (especially tiling)

import numpy as np
import warp as wp

import matplotlib.pyplot as plt

import sys
sys.path.append("./src")

from _benchmark import benchmark_functions

# Matplotlib global settings
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1/25.4

TILE_SIZE = wp.constant(1024)

@wp.kernel
def kernel_rowsum (
    a: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32)
):
    i, j = wp.tid()
    out[i, 0] += a[i, j]


@wp.kernel
def kernel_rowsum_tiled (
    a: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32)
):
    i = wp.tid()
    # Load entire row into shared memory
    t = wp.tile_load(a[i], TILE_SIZE)
    s = wp.tile_sum(t)
    # Store result
    wp.tile_store(out[i], s)


if __name__ == "__main__":
    ### Benchmark graph coloring performance scaling with mesh size
    NUM_SAMPLES = 10
    sweep_n = np.linspace(100, 100000, 20, dtype=int)
    metrics = {
        "n": [],
        "times": {
            "rowsum": [],
            "rowsum_tiled": [],
        },
    }
    for n in sweep_n:
        # Set up input
        a_np = np.arange(n, dtype=np.float32).reshape(-1, 1) * np.ones((1, TILE_SIZE), dtype=np.float32) / 100.0
        metrics["n"].append(n)

        def launch_rowsum (input_array: np.ndarray) -> np.ndarray:
            a = wp.array(input_array, dtype=wp.float32)
            out = wp.zeros((a.shape[0], 1), dtype=wp.float32)
            wp.launch(kernel_rowsum, dim=(a.shape[0], a.shape[1]), inputs=[a, out])
            return out.numpy()
        
        def launch_rowsum_tiled (input_array: np.ndarray) -> np.ndarray:
            a = wp.array(input_array, dtype=wp.float32)
            out = wp.zeros((a.shape[0], 1), dtype=wp.float32)
            wp.launch_tiled(kernel_rowsum_tiled, dim=(a.shape[0],), inputs=[a, out], block_dim=64)
            return out.numpy()
        
        times = benchmark_functions([launch_rowsum, launch_rowsum_tiled], a_np, num_samples=NUM_SAMPLES)
        metrics["times"]["rowsum"].append(times["launch_rowsum"])
        metrics["times"]["rowsum_tiled"].append(times["launch_rowsum_tiled"])
        print(f"Completed benchmark for n={n}.")
        
        ### Plot performance scaling
        fig, ax = plt.subplots(1, 1, figsize=(120*mm, 70*mm))

        for method in metrics["times"]:
            times_mean = [np.mean(t) for t in metrics["times"][method]]
            times_std = [np.std(t) for t in metrics["times"][method]]
            ax.plot(metrics["n"], times_mean, 'o-', label=method)
            ax.fill_between(metrics["n"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel("Number of Elements (-)")
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.grid()
        fig.suptitle("Row Sum Performance Scaling")

        fig.savefig("outputs/benchmark_rowsum.png", dpi=300, bbox_inches='tight')
        plt.close(fig)




