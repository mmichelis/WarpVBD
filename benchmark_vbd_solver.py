### Benchmark VBD solver performance scaling with mesh size.

import time
import numpy as np
import warp as wp

import matplotlib.pyplot as plt

from run_cantilever import CantileverSim
from benchmark import benchmark_functions

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
    num_samples = 5
    num_voxels = np.linspace(10, 50, 10, dtype=int)
    metrics = {
        "num_vertices": [],
        "num_elements": [],
        "times": {
            "vbd_solve": []
        },
    }
    for nx in num_voxels:
        dx = 1.0/nx
        sim = CantileverSim(
            nx=(nx, nx, nx),
            dx=(dx, dx, dx),
            density=1070, youngs_modulus=250e3, poissons_ratio=0.45,
            device="cuda"
        )
        metrics["num_vertices"].append(sim.initial_positions.shape[0])
        metrics["num_elements"].append(sim.elements.shape[0])

        ### Benchmark solver time
        timestep = 1e-3
        n_timesteps = 20
        
        def vbd_solve (simulation, nt, dt):
            simulation.reset()
            for _ in range(nt):
                simulation.step(dt)

        times = benchmark_functions([vbd_solve], sim, n_timesteps, timestep, num_samples=num_samples)
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





