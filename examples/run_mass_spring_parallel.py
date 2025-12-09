### Mass Spring System simulation falling under gravity.

import numpy as np
import warp as wp

import matplotlib.pyplot as plt
import argparse
import time, os, shutil

import sys
sys.path.append("./src")

from _utils import voxel2hex, hex2tets
from _renderer import PbrtRenderer, export_mp4, filter_elements
import vbd_solver

# Matplotlib global settings
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})
mm = 1/25.4


class MassSpringParallelSim:
    """
    Lots of Mass Spring System simulations in parallel, where the masses are standardized as a 1m x 1m x 1m cube of 1kg. Stiffnesses are varied between the simulations.
    """
    def __init__ (self, nsim=4, nx=7, density=1, youngs_modulus=6e3, poissons_ratio=0.45, dx_tol=1e-9, max_iter=1000, device="cuda"):
        ### Set up tetrahedral mesh
        dx = 1.0/nx
        self.nsim = nsim
        # Split parallel simulations into xy grid
        nsimy = int(np.ceil(np.sqrt(nsim)))
        nsimx = int(np.ceil(nsim / nsimy))
        padding_voxels = int(1.0*nx) # Padding between simulations

        ### Add a wall plane as a hex mesh box. Center should align with mass spring top center.
        voxels = np.zeros((nsimx*(nx+padding_voxels)-padding_voxels, nsimy*(nx+padding_voxels)-padding_voxels, 2*nx), dtype=bool)
        wall_voxels = np.zeros((nsimx*(nx+padding_voxels)-padding_voxels, nsimy*(nx+padding_voxels)-padding_voxels, 1), dtype=bool)
        for i in range(nsimx):
            for j in range(nsimy):
                sim_idx = i*nsimy + j
                if sim_idx >= nsim:
                    break
                x_start = i*(nx+padding_voxels)
                y_start = j*(nx+padding_voxels)
                voxels[x_start:x_start+nx, y_start:y_start+nx, :nx] = True
                voxels[x_start+nx//2:x_start+nx//2+1, y_start+nx//2:y_start+nx//2+1, nx:] = True
                wall_voxels[x_start:x_start+nx, y_start:y_start+nx, 0] = True

        vertices, hex_elements = voxel2hex(voxels, dx, dx, dx)
        self.wall_vertices, self.wall_elements = voxel2hex(wall_voxels, dx, dx, 0.1)
        self.wall_vertices[:,2] += 2*nx*dx
        print(f"Voxel shape: {voxels.shape}.")

        elements = hex2tets(hex_elements)
        n_vertices = vertices.shape[0]
        n_elements = elements.shape[0]
        self.initial_positions = vertices
        self.elements = elements
        print(f"\033[95mGenerated tetrahedral mesh with {n_vertices} vertices and {n_elements} elements.\033[0m")

        ### Assign varying stiffnesses for each simulation, and assign each vertex/element to a sim environment
        self.vertex_sim_env = -1 * np.ones((vertices.shape[0],), dtype=int)
        self.element_sim_env = -1 * np.ones((elements.shape[0],), dtype=int)
        youngs_moduli = np.zeros((n_elements,), dtype=np.float64)
        poissons_ratios = poissons_ratio * np.ones((n_elements,), dtype=np.float64)
        random_ym = np.random.uniform(-0.5, 0.5, size=(nsim,))
        for i in range(n_elements):
            # Determine which simulation this element belongs to
            com = vertices[elements[i]].mean(axis=0)
            sim_x = int(com[0] // ((nx+padding_voxels)*dx))
            sim_y = int(com[1] // ((nx+padding_voxels)*dx))
            sim_idx = sim_x * nsimy + sim_y
            assert sim_idx < nsim, "Element assigned to non-existing simulation!"

            self.element_sim_env[i] = sim_idx
            for v_idx in elements[i]:
                self.vertex_sim_env[v_idx] = sim_idx

            # Random Young's modulus between simulations
            youngs_moduli[i] = youngs_modulus * (1.0 + random_ym[sim_idx])
        print(f"\033[95mYoung's modulus range: {youngs_moduli.min()/1e3:.2f}kPa to {youngs_moduli.max()/1e3:.2f}kPa.\033[0m")

        ### Set active mask, tip indices, and assign each vertex to a sim environment
        active_mask = np.ones((vertices.shape[0],), dtype=bool)
        tip_idx = {i: [] for i in range(nsim)}
        for i in range(vertices.shape[0]):
            if vertices[i,2] > 2*nx*dx-1e-3:
                active_mask[i] = False
            elif vertices[i,2] < 1e-3:
                tip_idx[self.vertex_sim_env[i]].append(i)
        self.tip_idx = tip_idx
        print(f"Active vertices: {np.sum(active_mask)}/{vertices.shape[0]}")

        
        ### Initialize solver
        solution = wp.array(vertices, dtype=wp.vec3d, device=device)
        elements = wp.array(elements, dtype=wp.vec4i, device=device)
        active_mask = wp.array(active_mask, dtype=wp.bool, device=device)
        densities = wp.array(density * np.ones(n_elements), dtype=wp.float64, device=device)
        youngs_moduli = wp.array(youngs_moduli, dtype=wp.float64, device=device)
        poissons_ratios = wp.array(poissons_ratios, dtype=wp.float64, device=device)

        self.solver = vbd_solver.VBDSolver(
            initial_positions=solution,
            elements=elements,
            densities=densities,
            youngs_modulus=youngs_moduli,
            poissons_ratio=poissons_ratios,
            damping_coefficient=0.0,
            active_mask=active_mask,
            gravity=wp.vec3d(0.0, 0.0, -9.81),
            dx_tol=dx_tol,
            max_iter=max_iter,
            device=device
        )

        # Adjust camera based on number of simulations
        self.camera_pos = (-0.25-0.02*(nsimx//2), -1.0-0.02*(nsimy//2), 0.4)
        self.camera_lookat = (0.1*(0.25+nsimx*(nx+padding_voxels)*dx/2), 0.1*(0.5+nsimy*(nx+padding_voxels)*dx/2), 0.1)


    def step (self, dt):
        return self.solver.step(self.solver.old_positions, wp.float64(dt))

    def render (self, vertices, filename=None, spp=4):
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
            'camera_pos': self.camera_pos,
            'camera_lookat': self.camera_lookat,
            'resolution': (800, 800)
        }
        transforms=[
            ('s', 0.1),
            ('t', [0, 0, 0.15])
        ]
        renderer = PbrtRenderer(options)

        # Color each simulation differently
        color_map = plt.get_cmap("tab10")
        for sim_idx in range(self.nsim):
            v, e = filter_elements(vertices, self.elements, np.argwhere(self.element_sim_env == sim_idx).flatten())
            color = color_map(sim_idx % 10)
            hex_color = "{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

            renderer.add_tri_mesh(vertices=v, elements=e, render_edges=True, color=hex_color, transforms=transforms)

        renderer.add_tri_mesh(objFile='asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7', transforms=[('s', 4)])

        ### Add a wall plane as a hex mesh box.
        renderer.add_hex_mesh(vertices=self.wall_vertices, elements=self.wall_elements, color="484848", transforms=transforms)

        renderer.render()

    def reset (self):
        self.solver.reset()


def main (args):
    # Create output directories
    output_folder = "outputs/mass_spring_parallel"
    os.makedirs(output_folder, exist_ok=True)
    if args.render:
        if os.path.isdir(f"{output_folder}/frames"):
            shutil.rmtree(f"{output_folder}/frames")
        os.makedirs(f"{output_folder}/frames", exist_ok=False)

    # Set up simulation
    sim = MassSpringParallelSim(nsim=args.nsim, nx=args.nx, dx_tol=1e-5, device="cuda")

    ### Begin the solve
    n_seconds = 5.0
    fps = 50
    n_timesteps = int(n_seconds * fps)
    n_substeps = 5
    dt = 1/fps/n_substeps

    tip_positions = {i: [] for i in range(args.nsim)}
    for t in range(n_timesteps):
        start_time = time.time()

        for _ in range(n_substeps):
            solution = sim.step(dt)
            for i in range(args.nsim):
                tip_positions[i].append(solution.numpy()[sim.tip_idx[i]].mean(axis=0))

        end_time = time.time()
        print(f"---Timestep [{t:04d}/{n_timesteps}] ({1e3*dt*n_substeps:.1f}ms) in {1e3*(end_time - start_time):.3f}ms: Mean Positions: {solution.numpy().mean(axis=0)}")

        if args.render:
            sim.render(solution.numpy(), filename=f"{output_folder}/frames/mass_spring_parallel_{t:03d}.png", spp=4)

        # Plot Tip Displacements
        fig, ax = plt.subplots(figsize=(3,2))
        tip_positions_np = np.array(tip_positions)
        ax.plot(np.arange(len(tip_positions_np)) * dt, tip_positions_np[:, 2])  # Plot z displacement over time
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z Position (m)")
        ax.grid()
        ax.set_xlim(0, n_seconds)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        fig.savefig(f"{output_folder}/displacement_mass_spring_parallel.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Store as csv
    np.savetxt(f"{output_folder}/displacement_mass_spring_parallel.csv", np.array(tip_positions), delimiter=",")

    if args.render:
        # Render mp4
        export_mp4(f"{output_folder}/frames", f"{output_folder}/mass_spring_parallel.mp4", fps=fps)


if __name__ == "__main__":
    wp.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=6, help="Number of parallel simulations")
    parser.add_argument("--nx", type=int, default=7, help="Number of voxels in x direction")
    parser.add_argument("--dx", type=float, default=0.01, help="Voxel size in x direction")
    parser.add_argument("--render", action='store_true', help="Visualize option")
    args = parser.parse_args()
    
    main(args)