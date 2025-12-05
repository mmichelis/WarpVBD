
import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time

from _utils import voxel2hex, hex2tets
from _renderer import PbrtRenderer, export_mp4
from coloring import graph_coloring, compute_adjacency_dict
# import vbd_solver
import vbd_solver_np as vbd_solver


def render (vertices, elements, filename=None, spp=4):
    """
    Short rendering script for tetrahedral meshes using PBRT.
    """
    options = {
        'file_name': filename,
        'light_map': 'uffizi-large.exr',
        'sample': spp,
        'max_depth': 2,
        'camera_pos': (0.1, -0.75, 0.5),   # Position of camera
        'camera_lookat': (0.1, 0.25, 0.25),     # Position that camera looks at
    }
    transforms=[
        ('s', 1),
        ('t', [0, 0, 0.3])
    ]
    renderer = PbrtRenderer(options)

    renderer.add_tri_mesh(vertices=vertices, elements=elements, render_edges=True, color="496d8a", transforms=transforms)
    renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])
    renderer.render()


def main (args):
    ### Set up tetrahedral mesh
    voxels = np.ones((args.nx, args.ny, args.nz), dtype=bool)
    vertices, hex_elements = voxel2hex(voxels, args.dx, args.dy, args.dz)
    elements = hex2tets(hex_elements)
    points = [wp.vec3(point) for point in vertices]
    tet_indices = elements.flatten().tolist()
    print(f"Generated tetrahedral mesh with {len(points)} vertices and {len(elements)} elements.")

    # Set active mask
    active_mask = np.ones((vertices.shape[0],), dtype=bool)
    tip_idx = []
    for i in range(vertices.shape[0]):
        if vertices[i,0] < 1e-3:
            active_mask[i] = False
        elif vertices[i,0] > (args.nx * args.dx - 1e-3):
            tip_idx.append(i)
    # active_mask = np.ones((vertices.shape[0],), dtype=bool)
    print(f"Active vertices: {np.sum(active_mask)}/{vertices.shape[0]}")

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
    device = "cuda"
    wp.init()
    solution = wp.array(vertices, dtype=wp.vec3d, device=device)
    elements = wp.array(elements, dtype=wp.vec4i, device=device)
    adj_v2e = wp.array(adj_v2e, dtype=wp.int32, device=device)
    colors = wp.array(colors, dtype=wp.int32, device=device)
    active_mask = wp.array(active_mask, dtype=wp.bool, device=device)
    densities = wp.array(1070 * np.ones(num_elements), dtype=wp.float64, device=device)  # Uniform density
    n_seconds = 1.5
    fps = 100
    n_timesteps = int(n_seconds * fps)
    n_substeps = 10
    dt = 1/fps/n_substeps
    print(f"Simulating {n_seconds}s of dynamics in {n_timesteps} timesteps of {n_substeps} substeps each (dt={dt:.6f}s).")

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
    tip_positions = []
    for t in range(n_timesteps):
        start_time = time.time()

        for i in range(n_substeps):
            solution = solver.step(solution, wp.float64(dt))
            tip_positions.append(solution.numpy()[tip_idx].mean(axis=0))

        end_time = time.time()
        print(f"---Timestep [{t:04d}/{n_timesteps}] ({1e3*dt*n_substeps:.1f}ms) in {1e3*(end_time - start_time):.3f}ms: Mean Positions: {solution.numpy().mean(axis=0)}")

        if args.visualize:
            render(solution.numpy(), elements.numpy(), filename=f"outputs/sim/cantilever_{t:03d}.png", spp=4)

        # Plot Tip Displacements
        fig, ax = plt.subplots(figsize=(3,2))
        tip_positions_np = np.array(tip_positions)
        ax.plot(np.arange(len(tip_positions_np)) * dt, tip_positions_np[:, 2])  # Plot z displacement over time
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z Position (m)")
        ax.grid()
        ax.set_xlim(0, n_seconds)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        fig.savefig("outputs/tip_displacement.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    if args.visualize:
        # Render mp4
        export_mp4("outputs/sim/", "outputs/cantilever.mp4", fps=int(0.25*fps), name_prefix="cantilever_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=10, help="Number of voxels in x direction")
    parser.add_argument("--ny", type=int, default=3, help="Number of voxels in y direction")
    parser.add_argument("--nz", type=int, default=3, help="Number of voxels in z direction")
    parser.add_argument("--dx", type=float, default=0.01, help="Voxel size in x direction")
    parser.add_argument("--dy", type=float, default=None, help="Voxel size in y direction")
    parser.add_argument("--dz", type=float, default=None, help="Voxel size in z direction")
    parser.add_argument("--visualize", action='store_true', help="Visualize option")
    args = parser.parse_args()

    if args.dy is None:
        args.dy = args.dx
    if args.dz is None:
        args.dz = args.dx
    
    main(args)