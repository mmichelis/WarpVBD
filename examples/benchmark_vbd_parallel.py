### Benchmark VBD solver performance scaling with mesh size, but this time in terms of parallel mass spring simulations.

import numpy as np
import warp as wp

import matplotlib.pyplot as plt

import sys
sys.path.append("./src")

from run_mass_spring_parallel import MassSpringParallelSim
from _utils import benchmark_functions

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

    n_samples = 3
    nsims = [2**i for i in range(0, 13)]
    metrics = {
        "n_vertices": [],
        "n_simulations": [],
        "times": {
            "vbd_solve": []
        },
        "tip_positions": []
    }
    for nsim in nsims:
        sim = MassSpringParallelSim(nsim=nsim, nx=7, dx_tol=1e-5, device="cuda")
        metrics["n_vertices"].append(sim.initial_positions.shape[0])
        metrics["n_simulations"].append(nsim)

        ### Benchmark solver time
        timestep = 4e-3
        n_timesteps = 125
        
        def vbd_solve (simulation, nt, dt):
            simulation.reset()
            for _ in range(nt):
                simulation.step(dt)

        times = benchmark_functions([vbd_solve], sim, n_timesteps, timestep, num_samples=n_samples)
        metrics["times"]["vbd_solve"].append(times["vbd_solve"])
        
        ### Plot performance scaling
        fig, ax = plt.subplots(1, 1, figsize=(3,2))
        for method in metrics["times"]:
            times_mean = [np.mean(t)/(timestep*n_timesteps) for t in metrics["times"][method]]
            times_std = [np.std(t)/(timestep*n_timesteps) for t in metrics["times"][method]]
            ax.plot(metrics["n_vertices"], times_mean, 'o-', label=method)
            ax.fill_between(metrics["n_vertices"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel("Number of Vertices (-)")
        ax.set_ylabel("Time per Simulated Second (s)")
        ax.grid()
        fig.savefig("outputs/benchmark_runtime_parallel.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        ### Plot scaling of realtime factor w.r.t. number of simulations
        fig, ax = plt.subplots(1, 1, figsize=(3,2))
        for method in metrics["times"]:
            factor_mean = np.array(metrics["n_simulations"]) * np.array([timestep*n_timesteps/np.mean(t) for t in metrics["times"][method]])
            factor_std = np.array(metrics["n_simulations"]) * np.array([np.std((timestep*n_timesteps)/np.array(t)) for t in metrics["times"][method]])
            ax.plot(metrics["n_simulations"], factor_mean, 'o-', label=method)
            ax.fill_between(metrics["n_simulations"], np.array(factor_mean)-np.array(factor_std), np.array(factor_mean)+np.array(factor_std), alpha=0.3)
        # Draw a horizontal line at realtime factor = 1
        ax.axhline(y=1.0, color="gray", linestyle='--')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.set_xlabel("Number of Simulations (-)")
        ax.set_ylabel("Realtime Factor per Env (-)")
        ax.grid()
        fig.savefig("outputs/benchmark_realtime_parallel.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


        # Store metrics dictionary as npz
        np.savez_compressed("outputs/benchmark_parallel_metrics.npz", **metrics)




