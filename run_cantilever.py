
import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time

from _utils import voxel2hex, hex2tets
from _renderer import PbrtRenderer, export_mp4
import vbd_solver


class CantileverSim:
    def __init__ (self, nx=(10, 3, 3), dx=(0.01, 0.01, 0.01), density=1070, youngs_modulus=250e3, poissons_ratio=0.45, dx_tol=1e-9, device="cuda"):
        ### Set up tetrahedral mesh
        voxels = np.ones(nx, dtype=bool)
        vertices, hex_elements = voxel2hex(voxels, *dx)
        elements = hex2tets(hex_elements)
        n_vertices = vertices.shape[0]
        n_elements = elements.shape[0]
        self.elements = elements
        print(f"Generated tetrahedral mesh with {n_vertices} vertices and {n_elements} elements.")

        # Set active mask
        active_mask = np.ones((vertices.shape[0],), dtype=bool)
        tip_idx = []
        for i in range(vertices.shape[0]):
            if vertices[i,0] < 1e-3:
                active_mask[i] = False
            elif vertices[i,0] > (nx[0] * dx[0] - 1e-3):
                tip_idx.append(i)
        self.tip_idx = tip_idx
        print(f"Active vertices: {np.sum(active_mask)}/{vertices.shape[0]}")

        ### Initialize solver
        wp.init()
        solution = wp.array(vertices, dtype=wp.vec3d, device=device)
        elements = wp.array(elements, dtype=wp.vec4i, device=device)
        active_mask = wp.array(active_mask, dtype=wp.bool, device=device)
        densities = wp.array(density * np.ones(n_elements), dtype=wp.float64, device=device)  # Uniform density

        self.solver = vbd_solver.VBDSolver(
            initial_positions=solution,
            elements=elements,
            densities=densities,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
            damping_coefficient=0.0,
            active_mask=active_mask,
            gravity=wp.vec3d(0.0, 0.0, -9.81),
            dx_tol=dx_tol,
            device=device
        )

    def step (self, dt):
        return self.solver.step(self.solver.old_positions, wp.float64(dt))

    def render (self, vertices, color_groups=False, filename=None, spp=4):
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
            ('s', 2),
            ('t', [0, 0, 0.3])
        ]
        renderer = PbrtRenderer(options)

        renderer.add_tri_mesh(vertices=vertices, elements=self.elements, render_edges=True, color="496d8a", transforms=transforms)
        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])

        if color_groups:
            # Color each vertex as a sphere according to its color group
            n_colors = len(self.color_groups)
            cmap = plt.get_cmap('tab20', n_colors)
            for c, cg in enumerate(self.color_groups.values()):
                for idx in cg:
                    renderer.add_shape_mesh({'name': 'sphere', 'center': vertices[idx], 'radius': 0.005}, color=cmap(c)[:3], transforms=transforms)

        renderer.render()

    def reset (self):
        self.solver.reset()


def main (args):
    ### Set up tetrahedral mesh
    sim = CantileverSim(
        nx=(args.nx, args.ny, args.nz), 
        dx=(args.dx, args.dy, args.dz), 
        density=1070, youngs_modulus=250e3, poissons_ratio=0.49,
        dx_tol=1e-9, device="cuda"
    )

    ### Begin the solve
    n_seconds = 1.5
    fps = 100
    n_timesteps = int(n_seconds * fps)
    n_substeps = 10
    dt = 1/fps/n_substeps

    tip_positions = []
    for t in range(n_timesteps):
        start_time = time.time()

        for _ in range(n_substeps):
            solution = sim.step(dt)
            tip_positions.append(solution.numpy()[sim.tip_idx].mean(axis=0))

        end_time = time.time()
        print(f"---Timestep [{t:04d}/{n_timesteps}] ({1e3*dt*n_substeps:.1f}ms) in {1e3*(end_time - start_time):.3f}ms: Mean Positions: {solution.numpy().mean(axis=0)}")

        if args.visualize:
            sim.render(solution.numpy(), filename=f"outputs/sim/cantilever_{t:03d}.png", spp=4)

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

    # Store as csv
    np.savetxt("outputs/tip_positions.csv", tip_positions_np, delimiter=",")

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