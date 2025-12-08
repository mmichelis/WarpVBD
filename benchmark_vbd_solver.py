
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
    device = "cuda"
    wp.init()

    ### Benchmark VBD solver performance scaling with mesh size
    num_samples = 10
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
        num_vertices = vertices.shape[0]
        num_elements = elements.shape[0]
        metrics["num_vertices"].append(num_vertices)
        metrics["num_elements"].append(num_elements)
        print(f"Generated tetrahedral mesh with {num_vertices} vertices and {num_elements} elements.")

        ### Set active mask
        active_mask = np.ones((vertices.shape[0],), dtype=bool)
        tip_idx = []
        for i in range(vertices.shape[0]):
            if vertices[i,0] < 1e-3:
                active_mask[i] = False
            elif vertices[i,0] > (nx * dx - 1e-3):
                tip_idx.append(i)

        ### Perform graph coloring
        start_time = time.time()
        adjacency, vertex_valence = compute_adjacency_dict(elements)
        _, color_groups = graph_coloring(adjacency)
        end_time = time.time()
        print(f"Assigned {len(color_groups)} colors in {(end_time - start_time)*1000:.4f}ms. Vertex valence: {vertex_valence}")
        # Convert color groups to full array
        n_colors = len(color_groups)
        max_group_size = max(len(g) for g in color_groups.values())
        colors = np.full([n_colors, max_group_size], -1, dtype=int)
        for c in range(n_colors):
            group = color_groups[c]
            colors[c, :len(group)] = group

        # Find an adjacency mapping from vertex -> neighboring elements
        adj_v2e_list = [[] for _ in range(num_vertices)]
        max_incident_elements = 0
        for i, ele in enumerate(elements):
            for vertex in ele:
                adj_v2e_list[vertex].append(i)
                max_incident_elements = max(max_incident_elements, len(adj_v2e_list[vertex]))
        # Create fixed-size adjacency array
        adj_v2e = np.full((num_vertices, max_incident_elements), -1, dtype=int)
        for vertex in range(num_vertices):
            for j, ele_idx in enumerate(adj_v2e_list[vertex]):
                adj_v2e[vertex, j] = ele_idx


        ### Benchmark solver time
        solution = wp.array(vertices, dtype=wp.vec3d, device=device)
        elements = wp.array(elements, dtype=wp.vec4i, device=device)
        adj_v2e = wp.array(adj_v2e, dtype=wp.int32, device=device)
        colors = wp.array(colors, dtype=wp.int32, device=device)
        active_mask = wp.array(active_mask, dtype=wp.bool, device=device)
        densities = wp.array(1070 * np.ones(num_elements), dtype=wp.float64, device=device)
        dt = 1e-3
        n_timesteps = 100

        solver = vbd_solver.VBDSolver(
            initial_positions=solution,
            elements=elements,
            adj_v2e=adj_v2e,
            color_groups=colors,
            densities=densities,
            youngs_modulus=250e3,
            poisson_ratio=0.45,
            damping_coefficient=0.0,
            active_mask=active_mask,
            gravity=wp.vec3d(0.0, 0.0, -9.81),
            device=device
        )
        
        def vbd_solve (solver, initial_solution):
            solution = initial_solution
            solver.reset()
            for _ in range(n_timesteps):
                solution = solver.step(solution, wp.float64(dt))

        times = benchmark_functions([vbd_solve], solver, solution, num_samples=num_samples)
        metrics["times"]["vbd_solve"].append(times["vbd_solve"])

        
        ### Plot performance scaling
        fig, ax = plt.subplots(1, 1, figsize=(120*mm, 70*mm))
        for method in metrics["times"]:
            times_mean = [np.mean(t)/n_timesteps for t in metrics["times"][method]]
            times_std = [np.std(t)/n_timesteps for t in metrics["times"][method]]
            ax.plot(metrics["num_vertices"], times_mean, 'o-', label=method)
            ax.fill_between(metrics["num_vertices"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel("Number of Vertices (-)")
        ax.set_ylabel("Time per Step (s)")
        ax.legend()
        ax.grid()
        fig.suptitle("VBD Solve Performance Scaling")

        fig.savefig("outputs/benchmark_vbd_solve.png", dpi=300, bbox_inches='tight')
        plt.close(fig)




