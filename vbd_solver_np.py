### VBD solver implementation in Numpy without parallelization

import matplotlib.pyplot as plt

import numpy as np
import warp as wp

EPS = 1e-12
MAX_ELEMENTS_PER_VERTEX = 128

diljk = np.zeros([3,3,3,3])
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                diljk[i,j,k,l] = 1.0 if (i == l and j == k) else 0.0
diljk = diljk.reshape(9,9)

def inertial_gradient_hessian (
    vertex_idx,
    ele_idx,
    positions,
    old_positions,
    old_velocities,
    masses,
    gravity,
    dt
):
    x = positions[vertex_idx]
    x_prev = old_positions[vertex_idx]
    v_prev = old_velocities[vertex_idx]
    m = masses[ele_idx] / 4  # Mass of the vertex (1/4 of element mass)

    ### Assemble gradient and hessian
    gradient = np.zeros(3)
    hessian = np.zeros([3,3])

    # Kinetic
    y = x_prev + dt * v_prev
    gradient += m  / (dt * dt) * (x - y)
    hessian += m / (dt * dt) * np.eye(3)

    # Gravity
    gradient += -m * gravity
    # no hessian

    return gradient, hessian


def elastic_gradient_hessian (
    vertex_idx,
    ele_idx,
    positions,
    volumes,
    inv_Dm,
    dDs_dx,
    lame_mu,
    lame_lambda,
    elements
):
    ele = elements[ele_idx]
    inv_D = inv_Dm[ele_idx]

    ### Compute deformation gradient F
    x0 = positions[ele[0]]
    x1 = positions[ele[1]]
    x2 = positions[ele[2]]
    x3 = positions[ele[3]]

    # Deformed shape matrix # TODO write out matmul and skip mat33d initialization for performance
    Ds = np.stack([
        x0 - x3,
        x1 - x3,
        x2 - x3
    ], axis=1)
    F = Ds @ inv_D
    volume = volumes[ele_idx]

    ### Create derivatives of Ds after positions (constants)
    dF_dx = np.zeros([3,3,3]) # shape should technically be 9x3, but we can store as 3 mat33d for simplicity
    # Trick with masks to avoid branching
    for i in range(4):
        # Figure out which vertex we are differentiating with respect to
        # mask = wp.float64(vertex_idx == ele[i])
        if vertex_idx == ele[i]:
            # Mask will only be true for one of the four vertices, sum up contributions in x, y, z
            dF_dx[0] = dDs_dx[i][0] @ inv_D
            dF_dx[1] = dDs_dx[i][1] @ inv_D
            dF_dx[2] = dDs_dx[i][2] @ inv_D


    ### Assemble gradient and hessian
    gradient = np.zeros(3)
    hessian = np.zeros([3,3])

    # Elastic, St. Venant-Kirchhoff
    Ft = F.transpose()
    E = 0.5 * (Ft @ F - np.eye(3))
    Et = E.transpose()
    trE = np.trace(E)

    dE_dF = np.zeros([3,3,3,3])
    dEt_dF = np.zeros([3,3,3,3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == l:
                        dE_dF[i,j,k,l] += 0.5 * F[k, j]
                        dEt_dF[i,j,k,l] += 0.5 * F[j, k]
                    if j == l:
                        dE_dF[i,j,k,l] += 0.5 * F[i, k]
                        dEt_dF[i,j,k,l] += 0.5 * F[k, i]

    dphi_dF2 = (
        0.5 * lame_lambda * (
            0.5 * (F + Ft).reshape(9,1) @ (F + Ft).reshape(1,9) 
            + trE * (np.eye(9) + diljk)
        ) + lame_mu * (
            np.einsum('ka, lamn -> klmn', F, dE_dF).reshape(9,9)
            + np.einsum('bk, blmn -> klmn', F, dE_dF).reshape(9,9)
            + np.einsum('km, ln -> klmn', np.eye(3), E).reshape(9,9)
            + np.einsum('kn, ml -> klmn', np.eye(3), E).reshape(9,9)
        )
    )

    # Sum up all contributions
    for i in range(3):
        dEdxi = 0.5 * (Ft @ dF_dx[i] + dF_dx[i].transpose() @ F)
        gradient[i] = volume * (
            lame_lambda * np.trace(E) * np.trace(dEdxi)
            + 2 * lame_mu * np.trace(dEdxi.transpose() @ E)
        )
        gradient_ = volume * (
            0.5 * lame_lambda * trE * (F + Ft)
            + lame_mu * (F @ Et + Ft @ E)
        ).reshape(1,9) @ dF_dx[i].reshape(9,1)

        assert abs(gradient[i] - gradient_).max() < 1e-6, f"Gradient mismatch: {gradient[i]} vs {gradient_}"

        for j in range(3):
            dEdxj = 0.5 * (Ft @ dF_dx[j] + dF_dx[j].transpose() @ F)
            d2E_dxidxj = 0.5 * (dF_dx[i].transpose() @ dF_dx[j] + dF_dx[j].transpose() @ dF_dx[i])
            
            hessian[i, j] = volume * (
                lame_lambda * (np.trace(dEdxi) * np.trace(dEdxj) + np.trace(E) * np.trace(d2E_dxidxj))
                + 2 * lame_mu * (np.trace(dEdxi.transpose() @ dEdxj) + np.trace(d2E_dxidxj.transpose() @ E))
            )

            hessian_ = (volume * dF_dx[i].reshape(1,9) @ dphi_dF2 @ dF_dx[j].reshape(9,1))[0,0]

            # assert abs(hessian[i, j] - hessian_).max() < 1e0, f"Hessian mismatch: {hessian[i, j]} vs {hessian_}"
            if abs(hessian[i, j] - hessian_).max() > 1e0:
                print(f"Hessian mismatch: {hessian[i, j]} vs {hessian_}")

    return gradient, hessian


def accumulate_grad_hess (
    positions,
    old_positions,
    old_velocities,

    inv_Dm,
    dDs_dx,

    masses,
    volumes,
    lame_mu,
    lame_lambda,
    damping_coefficient,

    gravity,
    dt,

    elements,
    adj_v2e,
    color_groups,
    active_mask,

    gradients,
    hessians
):
    for c in range(color_groups.shape[0]):
        for i in range(color_groups.shape[1]):
            for j in range(adj_v2e.shape[1]):
                # Skip if this is a padding entry
                idx_v = color_groups[c, i]
                if idx_v == -1:
                    continue
                idx_e = adj_v2e[idx_v, j]
                if idx_e == -1:
                    continue
                
                # Skip if vertex is not active
                if not active_mask[idx_v]:
                    continue

                ### Compute gradients and hessians
                grad_inertial, hess_inertia = inertial_gradient_hessian(idx_v, idx_e, positions, old_positions, old_velocities, masses, gravity, dt)
                grad_elastic, hess_elastic = elastic_gradient_hessian(idx_v, idx_e, positions, volumes, inv_Dm, dDs_dx, lame_mu, lame_lambda, elements)
                # Add damping
                grad_damping = damping_coefficient * hess_elastic @ (positions[idx_v] - old_positions[idx_v]) / dt
                hess_damping = damping_coefficient * hess_elastic / dt

                ### Accumulate results
                for k in range(3):
                    gradients[idx_v, k] += grad_inertial[k] + grad_elastic[k] + grad_damping[k]
                    for l in range(3):
                        hessians[idx_v, k, l] += hess_inertia[k, l] + hess_elastic[k, l] + hess_damping[k, l]

    return gradients, hessians


def solve_grad_hess (
    gradients,
    hessians,
    active_mask,
    dx
):
    for i in range(gradients.shape[0]):
        # Skip if vertex is not active
        if not active_mask[i]:
            continue

        dx[i] = np.linalg.solve(hessians[i], -gradients[i])
        
    return dx


@wp.kernel
def position_initialization (
    positions: wp.array(dtype=wp.vec3d),
    velocities: wp.array(dtype=wp.vec3d),
    gravity: wp.vec3d,
    dt: wp.float64,
    active_mask: wp.array(dtype=wp.bool),
    
    new_positions: wp.array(dtype=wp.vec3d)
) -> None:
    i = wp.tid()

    if not active_mask[i]:
        return

    for j in range(3):
        new_positions[i][j] = positions[i][j] + velocities[i][j] * dt + gravity[j] * dt * dt


def compute_element_masses_volume (
    positions,
    elements,
    densities,

    masses,
    volumes
):
    for i in range(elements.shape[0]):
        ele = elements[i]
        density = densities[i]

        # Compute volume
        x0 = positions[ele[0]]
        x1 = positions[ele[1]]
        x2 = positions[ele[2]]
        x3 = positions[ele[3]]
        Dm = np.stack([
            x0 - x3,
            x1 - x3,
            x2 - x3
        ], axis=1)
        volume = abs(np.linalg.det(Dm)) / 6
        masses[i] = density * volume
        volumes[i] = volume

    return masses, volumes


def compute_inv_Dm (
    positions,
    elements,

    inv_Dm
):
    for i in range(elements.shape[0]):
        ele = elements[i]
        X0 = positions[ele[0]]
        X1 = positions[ele[1]]
        X2 = positions[ele[2]]
        X3 = positions[ele[3]]

        # Each column is an edge vector
        Dm = np.stack([
            X0 - X3,
            X1 - X3,
            X2 - X3
        ], axis=1)
        inv_Dm[i] = np.linalg.inv(Dm)

    return inv_Dm


class VBDSolver:
    def __init__(
            self, 
            initial_positions,
            elements,
            adj_v2e,
            color_groups,
            densities,
            lame_mu=0.0,
            lame_lambda=0.0,
            youngs_modulus=0.0,
            poisson_ratio=0.5,
            damping_coefficient=0.0,
            active_mask=None,
            gravity=np.array([0.0, 0.0, -9.81]),
            device=None
        ):
        self.initial_positions = initial_positions.numpy()
        self.old_positions = np.copy(self.initial_positions)
        self.old_velocities = np.zeros_like(self.initial_positions)
        self.elements = elements.numpy()
        self.adj_v2e = adj_v2e.numpy()
        self.color_groups = color_groups.numpy()
        self.densities = densities.numpy()
        self.damping_coefficient = damping_coefficient
        self.gravity = gravity
        self.device = device

        if active_mask is not None:
            self.active_mask = active_mask.numpy()
        else:
            self.active_mask = np.ones(self.initial_positions.shape[0])
    
        # If Lame parameters are not provided, compute them from Young's modulus and Poisson's ratio
        if (abs(lame_mu) == 0.0 and abs(lame_lambda) == 0.0) and (abs(youngs_modulus) == 0.0 or abs(poisson_ratio) >= 0.5):
            assert False, "Provide either Lame parameters or valid Young's modulus and Poisson's ratio!"
        elif abs(lame_mu) == 0.0 and abs(lame_lambda) == 0.0:
            lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_lambda = (youngs_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda


        ### Compute mass per element from element densities
        n_elements = elements.shape[0]
        self.masses = np.zeros(n_elements)
        self.volumes = np.zeros(n_elements)
        self.masses, self.volumes = compute_element_masses_volume(self.initial_positions, self.elements, self.densities, self.masses, self.volumes)

        ### Compute inverted undeformed/reference shape matrix for tetrahedrons.
        self.inv_Dm = np.zeros([n_elements, 3, 3])
        self.inv_Dm = compute_inv_Dm(self.initial_positions, self.elements, self.inv_Dm)

        # Precompute the constant dDs_dx
        dDs_dx = np.zeros([4, 3, 3, 3], dtype=np.float64)
        dDs_dx[0][0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[0][1] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[0][2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(3,3)

        dDs_dx[1][0] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[1][1] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[1][2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,3)

        dDs_dx[2][0] = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[2][1] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).reshape(3,3)
        dDs_dx[2][2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3,3)

        dDs_dx[3][0] = np.array([-1.0, -1.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]).reshape(3,3)
        dDs_dx[3][1] = np.array([-0.0, -0.0, -0.0, -1.0, -1.0, -1.0, -0.0, -0.0, -0.0]).reshape(3,3)
        dDs_dx[3][2] = np.array([-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -1.0, -1.0]).reshape(3,3)

        self.dDs_dx = dDs_dx


    def step (self, positions, dt):
        positions = positions.numpy()
        dt = float(dt)

        n_vertices = positions.shape[0]
        n_colors = self.color_groups.shape[0]
        n_vertices_per_color = self.color_groups.shape[1]
        n_elements_per_vertex = self.adj_v2e.shape[1]
        # Initial guess: explicit Euler
        new_positions = np.copy(positions)
        # wp.launch(
        #     position_initialization,
        #     dim=n_vertices,
        #     inputs=[positions, self.old_velocities, self.gravity, dt, self.active_mask],
        #     outputs=[new_positions]
        # )

        gradients = np.zeros((n_vertices, 3))
        hessians = np.zeros((n_vertices, 3, 3))

        hist_dx = []
        MAX_ITER = 100
        for i in range(MAX_ITER):
            gradients, hessians = accumulate_grad_hess(
                    new_positions, self.old_positions, self.old_velocities, 
                    self.inv_Dm, self.dDs_dx, 
                    self.masses, self.volumes, self.lame_mu, self.lame_lambda, self.damping_coefficient,
                    self.gravity, dt, 
                    self.elements, self.adj_v2e, self.color_groups, self.active_mask,
                    gradients, hessians
            )
            dx = np.zeros((n_vertices, 3))
            dx = solve_grad_hess(gradients, hessians, self.active_mask, dx)

            new_positions += dx
            print(abs(dx).max())
            # breakpoint()
            hist_dx.append(abs(dx).mean())
            if abs(dx).max() < 1e-6:
                break
        
        # breakpoint()
        if i == MAX_ITER - 1:
            print(f"Warning: VBD solver did not converge within the maximum number of iterations. Final dx max: {abs(dx).max()}")

        # print(f"Final dx in {i} iterations: {abs(dx.numpy()).max()}")

        # fig, ax = plt.subplots(figsize=(3,2))
        # ax.plot(np.array(hist_dx))
        # ax.set_xlabel("Iteration (-)")
        # ax.set_ylabel("Average dx (m)")
        # ax.grid()
        # ax.set_xlim(0, len(hist_dx))
        # ax.set_yscale("log")
        # fig.savefig("outputs/vbd_convergence.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)

        # Discretize velocities, implicit Euler
        self.old_velocities = (new_positions - self.old_positions) / dt
        # Set old positions for next velocities computation
        self.old_positions = np.copy(new_positions)

        # Check if nan because hessian singular TODO
        # print(f"All not nan: {wp.isnan(new_positions)}")

        return wp.array(new_positions)

