### Mass Spring System simulation falling under gravity.

import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time

from _utils import voxel2hex, hex2tets
from _renderer import PbrtRenderer, export_mp4
import vbd_solver


class MassSpringSim:
    """
    Mass Spring System Simulation, where the mass is standardized as a 1m x 1m x 1m cube of 1kg. 
    """
    def __init__ (self, nx=11, density=1, youngs_modulus=5e3, poissons_ratio=0.45, dx_tol=1e-9, max_iter=1000, device="cuda"):
        ### Set up tetrahedral mesh
        dx = 1.0/nx
        voxels = np.ones((nx, nx, 2*nx), dtype=bool)
        voxels[:,:,nx:] = False
        voxels[int(nx//2):int(nx//2)+1,int(nx//2):int(nx//2)+1,nx:] = True
        vertices, hex_elements = voxel2hex(voxels, dx, dx, dx)
        vertices[:,:2] -= vertices.mean(axis=0)[:2]
        elements = hex2tets(hex_elements)
        n_vertices = vertices.shape[0]
        n_elements = elements.shape[0]
        self.initial_positions = vertices
        self.elements = elements
        print(f"Generated tetrahedral mesh with {n_vertices} vertices and {n_elements} elements.")

        # Set active mask
        active_mask = np.ones((vertices.shape[0],), dtype=bool)
        tip_idx = []
        for i in range(vertices.shape[0]):
            if vertices[i,2] > 2*nx*dx-1e-3:
                active_mask[i] = False
            elif vertices[i,2] < 1e-3:
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
            max_iter=max_iter,
            device=device
        )

        ### Add a wall plane as a hex mesh box. Center should align with mass spring top center.
        com = vertices.mean(axis=0)
        wall_width, wall_depth, wall_height = 1.0, 1.0, 0.1
        wall_translation = np.array([com[0]-wall_width/2, com[1]-wall_depth/2, 2*nx*dx-wall_height/2])
        self.wall_vertices, self.wall_elements = voxel2hex(np.ones((1,1,1), dtype=bool), wall_width, wall_depth, wall_height)
        self.wall_vertices += wall_translation


    def step (self, dt):
        return self.solver.step(self.solver.old_positions, wp.float64(dt))

    def render (self, vertices, color_groups=False, filename=None, spp=4):
        """
        Rendering function using PBRT.

        Args:
            vertices (np.ndarray): Vertex positions to render.
            color_groups (bool): Whether to color vertices by groups.
            filename (str): Output filename for the rendered image.
            spp (int): Samples per pixel, defines how much noise is in the image. Lower is faster.
        """
        options = {
            'file_name': filename,
            'light_map': 'uffizi-large.exr',
            'sample': spp,
            'max_depth': 2,
            'camera_pos': (0.0, -0.75, 0.3),   # Position of camera
            'camera_lookat': (0.0, 0.25, 0.1),     # Position that camera looks at
        }
        transforms=[
            ('s', 0.1),
            ('t', [0, 0, 0.15])
        ]
        renderer = PbrtRenderer(options)

        renderer.add_tri_mesh(vertices=vertices, elements=self.elements, render_edges=True, color="496d8a", transforms=transforms)
        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])

        if color_groups:
            # Color each vertex as a sphere according to its color group
            n_colors = len(self.solver.color_groups)
            cmap = plt.get_cmap('tab20', n_colors)
            for c, cg in enumerate(self.solver.color_groups.numpy().values()):
                for idx in cg:
                    renderer.add_shape_mesh({'name': 'sphere', 'center': vertices[idx], 'radius': 0.005}, color=cmap(c)[:3], transforms=transforms)

        ### Add a wall plane as a hex mesh box.
        renderer.add_hex_mesh(vertices=self.wall_vertices, elements=self.wall_elements, color="2ca02c", transforms=transforms)

        renderer.render()

    def reset (self):
        self.solver.reset()


def main (args):
    # Set up simulation
    sim = MassSpringSim(nx=args.nx, dx_tol=1e-5, device="cuda")

    ### Begin the solve
    n_seconds = 2.0
    fps = 50
    n_timesteps = int(n_seconds * fps)
    n_substeps = 5
    dt = 1/fps/n_substeps

    tip_positions = []
    for t in range(n_timesteps):
        start_time = time.time()

        for _ in range(n_substeps):
            solution = sim.step(dt)
            tip_positions.append(solution.numpy()[sim.tip_idx].mean(axis=0))

        end_time = time.time()
        print(f"---Timestep [{t:04d}/{n_timesteps}] ({1e3*dt*n_substeps:.1f}ms) in {1e3*(end_time - start_time):.3f}ms: Mean Positions: {solution.numpy().mean(axis=0)}")

        if args.render:
            sim.render(solution.numpy(), filename=f"outputs/sim/mass_spring_{t:03d}.png", spp=4)

        # Plot Tip Displacements
        fig, ax = plt.subplots(figsize=(3,2))
        tip_positions_np = np.array(tip_positions)
        ax.plot(np.arange(len(tip_positions_np)) * dt, tip_positions_np[:, 2])  # Plot z displacement over time
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z Position (m)")
        ax.grid()
        ax.set_xlim(0, n_seconds)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        fig.savefig("outputs/displacement_mass_spring.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Store as csv
    np.savetxt("outputs/displacement_mass_spring.csv", tip_positions_np, delimiter=",")

    if args.render:
        # Render mp4
        export_mp4("outputs/sim/", "outputs/mass_spring.mp4", fps=fps, name_prefix="mass_spring_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=11, help="Number of voxels in x direction")
    parser.add_argument("--dx", type=float, default=0.01, help="Voxel size in x direction")
    parser.add_argument("--render", action='store_true', help="Visualize option")
    args = parser.parse_args()
    
    main(args)