
import numpy as np
import warp as wp

import argparse

from coloring import graph_coloring


def voxel2hex (voxels, dx, dy, dz):
    """
    Standard order of the hex vertices for vtk meshes or gmsh.
         5 -------- 6
        /|         /|
       4 -------- 7 |
       | |        | |
       | 1 -------|-2
       |/         |/
       0 -------- 3
    """
    nx, ny, nz = voxels.shape
    # One more vertex than cells in each direction
    vertex_flag = np.full((nx+1, ny+1, nz+1), -1, dtype=int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i][j][k]:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                vertex_flag[i + ii, j + jj, k + kk] = 0

    vertex_cnt = 0
    vertices = []
    for i in range(nx+1):
        for j in range(ny+1):
            for k in range(nz+1):
                if vertex_flag[i,j,k] == 0:
                    vertex_flag[i,j,k] = vertex_cnt
                    vertices.append((dx * i, dy * j, dz * k))
                    vertex_cnt += 1

    # Specific hexahedron face ordering
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i,j,k]:
                    elements.append([
                        vertex_flag[i,j,k],
                        vertex_flag[i,j+1,k],
                        vertex_flag[i+1,j+1,k],
                        vertex_flag[i+1,j,k],
                        vertex_flag[i,j,k+1],
                        vertex_flag[i,j+1,k+1],
                        vertex_flag[i+1,j+1,k+1],
                        vertex_flag[i+1,j,k+1],
                    ])

    vertices = np.stack(vertices, axis=0)
    elements = np.stack(elements, axis=0)

    return vertices, elements


def hex2tets (hex_elements):
    """
    Convert the hexahedrons defined above into tetrahedrons that have positive volume based on (x1-x0).dot((x2-x0)x(x3-x0)) > 0.
    Each hexahedron is split into 5 tetrahedrons.
    """
    n_hex = len(hex_elements)
    n_tet = 5 * n_hex

    tet_elements = np.zeros([n_tet, 4], dtype=int)
    for i in range(n_hex):
        tet_elements[5*i] = np.array([hex_elements[i, 0], hex_elements[i, 1], hex_elements[i, 4], hex_elements[i, 3]])
        tet_elements[5*i+1] = np.array([hex_elements[i, 1], hex_elements[i, 2], hex_elements[i, 6], hex_elements[i, 3]])
        tet_elements[5*i+2] = np.array([hex_elements[i, 1], hex_elements[i, 3], hex_elements[i, 6], hex_elements[i, 4]])
        tet_elements[5*i+3] = np.array([hex_elements[i, 1], hex_elements[i, 4], hex_elements[i, 6], hex_elements[i, 5]])
        tet_elements[5*i+4] = np.array([hex_elements[i, 3], hex_elements[i, 4], hex_elements[i, 7], hex_elements[i, 6]])
        
    return tet_elements


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


    colors = graph_coloring(elements)
    print(f"Assigned colors to elements: {colors}")