
import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import time

from _utils import voxel2hex, hex2tets
from coloring import graph_coloring, compute_adjacency_dict
from benchmark import benchmark_functions
import vbd_solver

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
    num_voxels = np.linspace(10, 50, 10, dtype=int)
    metrics = {
        "num_vertices": [],
        "num_elements": [],
        "times": {
            "vbd_solve": []
        },
    }
    for nx in num_voxels:
        ### Set up tetrahedral mesh
        voxels = np.ones((nx, nx, nx), dtype=bool)
        vertices, hex_elements = voxel2hex(voxels, dx, dx, dx)
        elements = hex2tets(hex_elements)
        points = [wp.vec3(point) for point in vertices]
        print(f"Generated tetrahedral mesh with {len(points)} vertices and {len(elements)} elements.")

        ### Perform graph coloring
        start_time = time.time()
        adjacency, maximal_degree = compute_adjacency_dict(elements)
        vertex_coloring, color_groups = graph_coloring(adjacency)
        end_time = time.time()
        print(f"Assigned {len(color_groups)} colors in {(end_time - start_time)*1000:.4f}ms. Maximal vertex degree: {maximal_degree}")
        # Convert color groups to full array
        n_colors = len(color_groups)
        max_group_size = max(len(g) for g in color_groups.values())
        colors = np.full([n_colors, max_group_size], -1, dtype=int)
        for c in range(n_colors):
            group = color_groups[c]
            colors[c, :len(group)] = group

        # Find an adjacency mapping from vertex -> neighboring elements
        num_vertices = vertices.shape[0]
        num_elements = elements.shape[0]
        adj_v2e = np.full((num_vertices, maximal_degree), -1, dtype=int)
        for i, ele in enumerate(elements):
            for vertex in ele:
                # Find the first available slot
                for j in range(maximal_degree):
                    if adj_v2e[vertex, j] == -1:
                        adj_v2e[vertex, j] = i
                        break
            
                assert j < maximal_degree, "Maximal degree exceeded!"


        metrics["num_vertices"].append(len(points))
        metrics["num_elements"].append(len(elements))

        ### Benchmark solver time
        solution = wp.array(vertices, dtype=wp.vec3d)
        elements = wp.array(elements, dtype=wp.vec4i)
        adj_v2e = wp.array(adj_v2e, dtype=wp.int32)
        colors = wp.array(colors, dtype=wp.int32)
        def vbd_solve (solution, elements, adj_v2e, colors):
            # Begin the solve
            n_timesteps = 1000
            dt = 1e-3
            for _ in range(n_timesteps):
                solution = vbd_solver.step(solution, elements, adj_v2e, colors, dt)

        times = benchmark_functions([vbd_solve], solution, elements, adj_v2e, colors, num_samples=num_samples)
    
        metrics["times"]["vbd_solve"].append(times["vbd_solve"])

        
        ### Plot performance scaling
        fig, ax = plt.subplots(1, 1, figsize=(120*mm, 70*mm))
        for method in metrics["times"]:
            times_mean = [np.mean(t) for t in metrics["times"][method]]
            times_std = [np.std(t) for t in metrics["times"][method]]
            ax.plot(metrics["num_vertices"], times_mean, 'o-', label=method)
            ax.fill_between(metrics["num_vertices"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel("Number of Vertices (-)")
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.grid()
        fig.suptitle("VBD Solve Performance Scaling")

        fig.savefig("outputs/benchmark_vbd_solve.png", dpi=300, bbox_inches='tight')
        plt.close(fig)




