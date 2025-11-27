
import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time

from _utils import voxel2hex, hex2tets
from coloring import graph_coloring


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=1, help="Number of voxels in x direction")
    parser.add_argument("--ny", type=int, default=1, help="Number of voxels in y direction")
    parser.add_argument("--nz", type=int, default=1, help="Number of voxels in z direction")
    parser.add_argument("--dx", type=float, default=1.0, help="Voxel size in x direction")
    parser.add_argument("--dy", type=float, default=None, help="Voxel size in y direction")
    parser.add_argument("--dz", type=float, default=None, help="Voxel size in z direction")
    args = parser.parse_args()

    if args.dy is None:
        args.dy = args.dx
    if args.dz is None:
        args.dz = args.dx
    
    ### Set up tetrahedral mesh
    voxels = np.ones((args.nx, args.ny, args.nz), dtype=bool)
    vertices, hex_elements = voxel2hex(voxels, args.dx, args.dy, args.dz)
    elements = hex2tets(hex_elements)
    points = [wp.vec3(point) for point in vertices]
    tet_indices = elements.flatten().tolist()
    print(f"Generated tetrahedral mesh with {len(points)} vertices and {len(elements)} elements.")

    start_time = time.time()
    colors = graph_coloring(elements)
    end_time = time.time()
    print(f"Assigned {colors.max() + 1} colors in {(end_time - start_time)*1000:.4f}ms.")

    ### Plot coloring in 3D
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