### Run standard cantilever oscillations with varying solver tolerance.

import time
import numpy as np
import warp as wp

import matplotlib.pyplot as plt

import sys
sys.path.append("./src")

from run_cantilever import CantileverSim
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

    ### Benchmark VBD solver performance scaling with mesh size
    num_samples = 3
    tolerances = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    metrics = {
        "tolerance": [],
        "times": {
            "vbd_solve": []
        },
        "tip_positions": []
    }
    for dx_tol in tolerances:
        metrics["tolerance"].append(dx_tol)
        sim = CantileverSim(
            density=1070, youngs_modulus=250e3, poissons_ratio=0.45,
            dx_tol=dx_tol, max_iter=2000, device="cuda"
        )

        ### Benchmark solver time
        timestep = 1e-3
        n_timesteps = 1000
        

        tip_positions = []
        def vbd_solve (simulation, nt, dt):
            simulation.reset()
            tip_positions.clear()
            for _ in range(nt):
                print(f"Timestep {_+1}/{nt} for dx_tol={dx_tol}", end="\r")
                solution = simulation.step(dt)
                tip_positions.append(solution.numpy()[simulation.tip_idx].mean(axis=0))

        times = benchmark_functions([vbd_solve], sim, n_timesteps, timestep, num_samples=num_samples)
        metrics["times"]["vbd_solve"].append(times["vbd_solve"])
        metrics["tip_positions"].append(np.array(tip_positions))
        
        ### Plot performance scaling
        fig, ax = plt.subplots(1, 1, figsize=(3,2))
        for method in metrics["times"]:
            times_mean = [np.mean(t)/(timestep*n_timesteps) for t in metrics["times"][method]]
            times_std = [np.std(t)/(timestep*n_timesteps) for t in metrics["times"][method]]
            ax.plot(metrics["tolerance"], times_mean, 'o-', label=method)
            ax.fill_between(metrics["tolerance"], np.array(times_mean)-np.array(times_std), np.array(times_mean)+np.array(times_std), alpha=0.3)
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel(r"$\Delta$x Tolerance (m)")
        ax.set_ylabel("Time per Simulated Second (s)")
        # ax.legend()
        ax.grid()
        # fig.suptitle("VBD Solve Tolerance Scaling")
        fig.savefig("outputs/benchmark_runtime_tolerance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


        ### Plot Tip Displacements
        fig, ax = plt.subplots(1, 1, figsize=(3,2))
        time_axis = np.arange(n_timesteps) * timestep
        for i, tol in enumerate(metrics["tolerance"]):
            ax.plot(time_axis, metrics["tip_positions"][i][:, 2], label=f"dx_tol={tol:.0e}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z Position (m)")
        ax.grid()
        ax.set_xlim(0, time_axis[-1])
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.legend(loc="lower center", bbox_to_anchor=[0.5, -0.5], fontsize=7, ncol=3, fancybox=True, handlelength=0.75, columnspacing=1.25, handletextpad=0.5)
        fig.savefig("outputs/benchmark_accuracy_tolerance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)



