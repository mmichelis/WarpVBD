
import numpy as np
import warp as wp

import matplotlib.pyplot as plt

import sys
sys.path.append("./src")

from _utils import voxel2hex, hex2tets, benchmark_functions
from _coloring import graph_coloring, graph_coloring_wp

# Matplotlib global settings
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1/25.4


if __name__ == "__main__":
    ### Benchmark graph coloring performance scaling with mesh size
    num_samples = 5
    dx = 1.0
    num_voxels = np.linspace(5, 20, 5, dtype=int)
    metrics = {
        "num_vertices": [],
        "num_elements": [],
        "times": {
            "numpy": [],
            "parallel": []
        },
    }
    for nx in num_voxels:
        ### Set up tetrahedral mesh
        voxels = np.ones((nx, nx, nx), dtype=bool)
        vertices, hex_elements = voxel2hex(voxels, dx, dx, dx)
        elements = hex2tets(hex_elements)
        points = [wp.vec3(point) for point in vertices]
        tet_indices = elements.flatten().tolist()
        print(f"Generated tetrahedral mesh with {len(points)} vertices and {len(elements)} elements.")

        metrics["num_vertices"].append(len(points))
        metrics["num_elements"].append(len(elements))

        times = benchmark_functions([graph_coloring, graph_coloring_wp], elements, num_samples=num_samples)
    
        metrics["times"]["numpy"].append(times["graph_coloring"])
        metrics["times"]["parallel"].append(times["graph_coloring_wp"])

        
        ### Plot performance scaling
        fig, axs = plt.subplots(1, 2, figsize=(200*mm, 70*mm))
        fig.subplots_adjust(wspace=0.4)

        for method in metrics["times"]:
            times_mean = [np.mean(t) for t in metrics["times"][method]]
            times_std = [np.std(t) for t in metrics["times"][method]]
            axs[0].plot(metrics["num_elements"], times_mean, 'o-', label=method.capitalize())
            axs[0].fill_between(metrics["num_elements"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel("Number of Elements (-)")
        axs[0].set_ylabel("Time (s)")
        axs[0].legend()
        axs[0].grid()

        for method in metrics["times"]:
            times_mean = [np.mean(t) for t in metrics["times"][method]]
            times_std = [np.std(t) for t in metrics["times"][method]]
            axs[1].plot(metrics["num_vertices"], times_mean, 'o-', label=method.capitalize())
            axs[1].fill_between(metrics["num_vertices"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel("Number of Vertices (-)")
        axs[1].set_ylabel("Time (s)")
        axs[1].legend()
        axs[1].grid()
        fig.suptitle("Graph Coloring Performance Scaling")

        fig.savefig("outputs/benchmark_graph_coloring.png", dpi=300, bbox_inches='tight')
        plt.close(fig)




