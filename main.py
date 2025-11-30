
import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time

from _utils import voxel2hex, hex2tets
from coloring import graph_coloring, compute_adjacency_dict
import vbd_solver



def main (args):
    ### Set up tetrahedral mesh
    voxels = np.ones((args.nx, args.ny, args.nz), dtype=bool)
    vertices, hex_elements = voxel2hex(voxels, args.dx, args.dy, args.dz)
    elements = hex2tets(hex_elements)
    points = [wp.vec3(point) for point in vertices]
    tet_indices = elements.flatten().tolist()
    print(f"Generated tetrahedral mesh with {len(points)} vertices and {len(elements)} elements.")

    ### Perform graph coloring
    start_time = time.time()
    adjacency, maximal_degree = compute_adjacency_dict(elements)
    vertex_coloring, color_groups = graph_coloring(adjacency)
    end_time = time.time()
    print(f"Assigned {len(color_groups)} colors in {(end_time - start_time)*1000:.4f}ms.")
    print(f"Maximal vertex degree: {maximal_degree}")
    # Convert color groups to full array
    n_colors = len(color_groups)
    max_group_size = max(len(g) for g in color_groups.values())
    colors = np.full([n_colors, max_group_size], -1, dtype=int)
    for c in range(n_colors):
        group = color_groups[c]
        colors[c, :len(group)] = group

    ### Plot coloring in 3D
    if args.visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ele in elements:
            ax.plot(vertices[ele[[0,1]],0], vertices[ele[[0,1]],1], vertices[ele[[0,1]],2], 'k-', linewidth=1.0, alpha=0.8)
            ax.plot(vertices[ele[[0,2]],0], vertices[ele[[0,2]],1], vertices[ele[[0,2]],2], 'k-', linewidth=1.0, alpha=0.8)
            ax.plot(vertices[ele[[0,3]],0], vertices[ele[[0,3]],1], vertices[ele[[0,3]],2], 'k-', linewidth=1.0, alpha=0.8)
            ax.plot(vertices[ele[[1,2]],0], vertices[ele[[1,2]],1], vertices[ele[[1,2]],2], 'k-', linewidth=1.0, alpha=0.8)
            ax.plot(vertices[ele[[1,3]],0], vertices[ele[[1,3]],1], vertices[ele[[1,3]],2], 'k-', linewidth=1.0, alpha=0.8)
            ax.plot(vertices[ele[[2,3]],0], vertices[ele[[2,3]],1], vertices[ele[[2,3]],2], 'k-', linewidth=1.0, alpha=0.8)
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=colors, cmap='jet', s=50, alpha=1.0)
        ax.set_title("Graph Coloring of Tetrahedral Mesh Vertices")
        fig.savefig("graph_coloring.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

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

    ### Begin the solve
    solution = wp.array(vertices, dtype=wp.vec3d)
    elements = wp.array(elements, dtype=wp.vec4i)
    adj_v2e = wp.array(adj_v2e, dtype=wp.int32)
    colors = wp.array(colors, dtype=wp.int32)
    masses = wp.array(np.ones((num_vertices,), dtype=np.float64), dtype=wp.float64)  # Uniform masses for simplicity
    n_timesteps = 10
    dt = 1e-2

    solver = vbd_solver.VBDSolver(
        initial_positions=solution,
        elements=elements,
        adj_v2e=adj_v2e,
        color_groups=colors,
        masses=masses
    )
    for t in range(n_timesteps):
        solution = solver.step(solution, wp.float64(dt))
        print(f"Timestep {t*dt:.4f}: Mean Positions: {solution.numpy().mean(axis=0)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=1, help="Number of voxels in x direction")
    parser.add_argument("--ny", type=int, default=1, help="Number of voxels in y direction")
    parser.add_argument("--nz", type=int, default=1, help="Number of voxels in z direction")
    parser.add_argument("--dx", type=float, default=1.0, help="Voxel size in x direction")
    parser.add_argument("--dy", type=float, default=None, help="Voxel size in y direction")
    parser.add_argument("--dz", type=float, default=None, help="Voxel size in z direction")
    parser.add_argument("--visualize", action='store_true', help="Visualize option")
    args = parser.parse_args()

    if args.dy is None:
        args.dy = args.dx
    if args.dz is None:
        args.dz = args.dx
    
    main(args)