### VBD solver implementation

import time
import matplotlib.pyplot as plt

import numpy as np
import warp as wp

from coloring import graph_coloring, compute_adjacency_dict

EPS = 1e-12
MAX_ELEMENTS_PER_VERTEX = 128  # Adjust as needed for maximum number of elements incident to a vertex


@wp.func
def inertial_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    old_positions: wp.array(dtype=wp.vec3d),
    old_velocities: wp.array(dtype=wp.vec3d),
    masses: wp.array(dtype=wp.float64),
    gravity: wp.vec3d,
    dt: wp.float64,
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the inertia potential with respect to vertex positions of a specific element. We add gravity as external acceleration here as well.
    """
    x = positions[vertex_idx]
    x_prev = old_positions[vertex_idx]
    v_prev = old_velocities[vertex_idx]
    m = masses[ele_idx] / wp.float64(4.0)  # Mass of the vertex (1/4 of element mass)

    ### Assemble gradient and hessian
    gradient = wp.vec3d()
    hessian = wp.mat33d()

    # Kinetic
    y = x_prev + dt * v_prev
    gradient += m  / (dt * dt) * (x - y)
    hessian += m / (dt * dt) * wp.identity(3, dtype=wp.float64)

    # Gravity
    gradient += -m * gravity
    # no hessian

    return gradient, hessian


@wp.func
def elastic_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    volumes: wp.array(dtype=wp.float64),
    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),
    lame_mus: wp.array(dtype=wp.float64),
    lame_lambdas: wp.array(dtype=wp.float64),
    elements: wp.array(dtype=wp.vec4i)
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
    ele = elements[ele_idx]
    inv_D = inv_Dm[ele_idx]
    lame_mu = lame_mus[ele_idx]
    lame_lambda = lame_lambdas[ele_idx]

    ### Compute deformation gradient F
    x0 = positions[ele[0]]
    x1 = positions[ele[1]]
    x2 = positions[ele[2]]
    x3 = positions[ele[3]]

    # Deformed shape matrix
    Ds = wp.matrix_from_cols(
        x0 - x3,
        x1 - x3,
        x2 - x3
    )
    F = Ds * inv_D
    volume = volumes[ele_idx]

    ### Create derivatives of Ds after positions (constants)
    dF_dx = wp.zeros(shape=(3,), dtype=wp.mat33d) # shape should technically be 9x3, but we can store as 3 mat33d for simplicity
    # Trick with masks to avoid branching
    for i in range(4):
        # Figure out which vertex we are differentiating with respect to
        mask = wp.float64(vertex_idx == ele[i])
        # Mask will only be true for one of the four vertices, sum up contributions in x, y, z
        dF_dx[0] += mask * dDs_dx[i][0] * inv_D
        dF_dx[1] += mask * dDs_dx[i][1] * inv_D
        dF_dx[2] += mask * dDs_dx[i][2] * inv_D


    ### Assemble gradient and hessian
    gradient = wp.vec3d()
    hessian = wp.mat33d()

    J = wp.determinant(F)
    Ft = wp.transpose(F)
    Ic = wp.trace(Ft * F)
    Finv = wp.inverse(F)
    FinvT = wp.transpose(Finv)

    ### Stable Neo-Hookean
    mu = wp.float64(4.0) * lame_mu / wp.float64(3.0)                    # Adjusted mu for stable Neo-Hookean
    lmbda = lame_lambda + wp.float64(5.0) * lame_mu / wp.float64(6.0)   # Adjusted lambda for stable Neo-Hookean
    alpha = wp.float64(1.0) + wp.float64(0.75) * mu / lmbda
    dPhi_dF = (
        mu * F * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) 
        + lmbda * (J - alpha) * J * FinvT
    )

    for i in range(3):
        gradient[i] = volume * wp.trace(wp.transpose(dPhi_dF) * dF_dx[i])
        for j in range(3):
            hessian[i, j] = volume * (
                wp.float64(2.0) * mu / ((Ic + wp.float64(1.0))*(Ic + wp.float64(1.0))) * wp.trace(Ft * dF_dx[i]) * wp.trace(Ft * dF_dx[j])
                + mu * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) * wp.trace(wp.transpose(dF_dx[i]) * dF_dx[j])
                + lmbda * J * (wp.float64(2.0) * J - alpha) * wp.trace(Finv * dF_dx[i]) * wp.trace(Finv * dF_dx[j])
                - lmbda * J * (J - alpha) * wp.trace(Finv * dF_dx[j] * Finv * dF_dx[i])
            )


    ### St. Venant-Kirchhoff
    # E = wp.float64(0.5) * (Ft * F - wp.identity(3, dtype=wp.float64))
    # for i in range(3):
    #     dEdxi = wp.float64(0.5) * (Ft * dF_dx[i] + wp.transpose(dF_dx[i]) * F)

    #     gradient[i] = volume * (
    #         lame_lambda * wp.trace(E) * wp.trace(dEdxi)
    #         + wp.float64(2.0) * lame_mu * wp.trace(wp.transpose(dEdxi) * E)
    #     )
        
    #     for j in range(3):
    #         dEdxj = wp.float64(0.5) * (Ft * dF_dx[j] + wp.transpose(dF_dx[j]) * F)
    #         d2E_dxidxj = wp.float64(0.5) * (wp.transpose(dF_dx[i]) * dF_dx[j] + wp.transpose(dF_dx[j]) * dF_dx[i])
            
    #         hessian[i, j] += volume * (
    #             lame_lambda * (wp.trace(dEdxi) * wp.trace(dEdxj) + wp.trace(E) * wp.trace(d2E_dxidxj))
    #             + wp.float64(2.0) * lame_mu * (wp.trace(wp.transpose(dEdxi) * dEdxj) + wp.trace(wp.transpose(d2E_dxidxj) * E))
    #         )

    return gradient, hessian


@wp.kernel
def accumulate_grad_hess (
    positions: wp.array(dtype=wp.vec3d),
    old_positions: wp.array(dtype=wp.vec3d),
    old_velocities: wp.array(dtype=wp.vec3d),

    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),

    masses: wp.array(dtype=wp.float64),
    volumes: wp.array(dtype=wp.float64),
    lame_mus: wp.array(dtype=wp.float64),
    lame_lambdas: wp.array(dtype=wp.float64),
    damping_coefficient: wp.float64,

    gravity: wp.vec3d,
    dt: wp.float64,

    elements: wp.array(dtype=wp.vec4i),
    adj_v2e: wp.array2d(dtype=wp.int32),
    color_group: wp.array(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.bool),

    gradients: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64)
) -> None:
    """
    Accumulate gradients and Hessians for each element neighboring a vertex, and solve the local linear system to get position updates. Done for all vertices in a color group.

    Inputs:
        positions: Current vertex positions.
        old_positions: Previous vertex positions.
        old_velocities: Previous vertex velocities.

        inv_Dm: Inverted undeformed shape matrices for elements.
        dDs_dx: Derivatives of Ds with respect to positions. Shape [4, 3, 3, 3], a 3x3 matrix for each vertex of the tetrahedron.

        masses: Element masses.
        volumes: Element volumes.
        lame_mus: Lame parameter mu.
        lame_lambdas: Lame parameter lambda.
        damping_coefficient: Damping coefficient.

        gravity: Gravity vector.
        dt: Time step size.

        elements: Mesh elements.
        adj_v2e: Adjacency mapping from vertex to neighboring elements.
        color_group: All vertices in the current color group.
        active_mask: Mask indicating active vertices.
    
    Outputs:
        new_positions: Updated vertex positions.
        grads: Accumulated gradients (for debugging).
        dxs: Position updates that were applied.
    """
    i, j = wp.tid()

    idx_v = color_group[i]
    if idx_v == -1:
        return

    # Skip if vertex is not active
    if not active_mask[idx_v]:
        return

    idx_e = adj_v2e[idx_v, j]
    if idx_e == -1:
        return

    ### Compute gradients and hessians
    grad_inertial, hess_inertia = inertial_gradient_hessian(idx_v, idx_e, positions, old_positions, old_velocities, masses, gravity, dt)
    grad_elastic, hess_elastic = elastic_gradient_hessian(idx_v, idx_e, positions, volumes, inv_Dm, dDs_dx, lame_mus, lame_lambdas, elements)
    # Add damping
    grad_damping = damping_coefficient * hess_elastic * (positions[idx_v] - old_positions[idx_v]) / dt
    hess_damping = damping_coefficient * hess_elastic / dt

    ### Accumulate results
    grad = grad_inertial + grad_elastic + grad_damping
    hess = hess_inertia + hess_elastic + hess_damping

    for k in range(3):
        gradients[idx_v, j, k] = grad[k]
        for l in range(3):
            hessians[idx_v, j, k, l] = hess[k, l]



@wp.kernel
def solve_grad_hess (
    gradients: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64),
    active_mask: wp.array(dtype=wp.bool),
    color_group: wp.array(dtype=wp.int32),

    dxs: wp.array2d(dtype=wp.float64)
) -> None:
    i = wp.tid()

    idx_v = color_group[i]
    if idx_v == -1:
        return

    # Skip if vertex is not active
    if not active_mask[idx_v]:
        return
    
    grad = wp.tile_load(gradients[idx_v], (MAX_ELEMENTS_PER_VERTEX, 3))
    hess = wp.tile_load(hessians[idx_v], (MAX_ELEMENTS_PER_VERTEX, 3, 3))
    # Sum up contributions of elements to vertices
    total_grad = wp.tile_sum(grad, axis=0)
    total_hess = wp.tile_sum(hess, axis=0)

    # Solve for dx
    L = wp.tile_cholesky(total_hess)
    out = wp.tile_cholesky_solve(L, -total_grad)
    
    # Store results
    wp.tile_store(dxs[idx_v], out)


@wp.kernel
def add_dx (
    positions: wp.array(dtype=wp.vec3d),
    dx: wp.array2d(dtype=wp.float64),
    active_mask: wp.array(dtype=wp.bool),
    color_group: wp.array(dtype=wp.int32),

    new_positions: wp.array(dtype=wp.vec3d)
) -> None:
    i = wp.tid()
    idx_v = color_group[i]
    if idx_v == -1:
        return
    if not active_mask[idx_v]:
        return

    for j in range(3):
        new_positions[idx_v][j] = positions[idx_v][j] + dx[idx_v][j]


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

    new_positions[i] = positions[i] + velocities[i] * dt + gravity * dt * dt


@wp.kernel
def compute_element_invDm_masses_volume (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    densities: wp.array(dtype=wp.float64),

    inv_Dm: wp.array(dtype=wp.mat33d),
    masses: wp.array(dtype=wp.float64),
    volumes: wp.array(dtype=wp.float64)
) -> None:
    """
    Compute mass and volume per element from element densities.

    Args:
        positions: Vertex positions.
        elements: Mesh elements.
        densities: Element densities.

    Outputs:
        masses: Output array for masses.
        volumes: Output array for volumes.
    """
    i = wp.tid()

    ele = elements[i]
    density = densities[i]

    # Compute volume
    x0 = positions[ele[0]]
    x1 = positions[ele[1]]
    x2 = positions[ele[2]]
    x3 = positions[ele[3]]
    Dm = wp.matrix_from_cols(
        x0 - x3,
        x1 - x3,
        x2 - x3
    )
    inv_Dm[i] = wp.inverse(Dm)
    volumes[i] = wp.abs(wp.determinant(Dm)) / wp.float64(6.0)
    masses[i] = density * volumes[i]

    # Check if inversion was successful
    res = Dm * inv_Dm[i]
    for r in range(3):
        for c in range(3):
            if r == c:
                assert wp.abs(res[r,c] - wp.float64(1.0)) < EPS, "Singular matrix encountered in inv_Dm computation!"
            else:
                assert wp.abs(res[r,c]) < EPS, "Singular matrix encountered in inv_Dm computation!"

@wp.kernel
def convert_youngs_poisson_to_lame (
    youngs_modulus: wp.array(dtype=wp.float64),
    poissons_ratio: wp.array(dtype=wp.float64),

    lame_mus: wp.array(dtype=wp.float64),
    lame_lambdas: wp.array(dtype=wp.float64)
) -> None:
    """
    Convert Young's modulus and Poisson's ratio to Lame parameters mu and lambda.
    """
    i = wp.tid()

    E = youngs_modulus[i]
    nu = poissons_ratio[i]

    lame_mus[i] = E / (wp.float64(2.0) * (wp.float64(1.0) + nu))
    lame_lambdas[i] = (E * nu) / ((wp.float64(1.0) + nu) * (wp.float64(1.0) - wp.float64(2.0) * nu))
    


class VBDSolver:
    def __init__(
            self, 
            initial_positions: wp.array(dtype=wp.vec3d),
            elements: wp.array(dtype=wp.vec4i),

            densities: wp.array(dtype=wp.float64),
            youngs_modulus: wp.array(dtype=wp.float64),
            poissons_ratio: wp.array(dtype=wp.float64),
            damping_coefficient: wp.float64 = wp.float64(1.0),

            active_mask: wp.array(dtype=wp.bool)=None,
            gravity: wp.vec3d = wp.vec3d(0.0, 0.0, -9.81),

            dx_tol: float=1e-9,
            max_iter: int=100,

            device: str = "cpu"
        ) -> None:
        """
        Initialize the VBD solver with the tetrahedral mesh and simulation parameters. Creates graph coloring and adjacency structures.

        Args:
            initial_positions: Initial vertex positions.
            elements: Mesh elements.

            densities: Element densities.
            youngs_modulus: Young's modulus.
            poissons_ratio: Poisson's ratio.

            gravity: Gravity vector.
            active_mask: Optional mask to indicate active vertices.
            dx_tol: Convergence tolerance for position updates.
            max_iter: Maximum number of solver iterations per time step.
            device: Device to run the simulation on ("cpu" or "cuda").
        """
        self.initial_positions = initial_positions
        self.old_positions = wp.clone(initial_positions) # For kinetic energy
        self.old_velocities = wp.zeros_like(initial_positions)
        self.elements = elements

        self.densities = densities
        self.damping_coefficient = damping_coefficient
        self.gravity = gravity
        self.dx_tol = dx_tol
        self.max_iter = max_iter
        self.device = device

        if active_mask is not None:
            self.active_mask = active_mask
        else:
            self.active_mask = wp.ones(initial_positions.shape[0], dtype=wp.bool, device=device)
    
        # If Lame parameters are not provided, compute them from Young's modulus and Poisson's ratio
        # if (wp.abs(lame_mu) == 0.0 and wp.abs(lame_lambda) == 0.0) and (wp.abs(youngs_modulus) == 0.0 or wp.abs(poissons_ratio) >= 0.5):
        #     assert False, "Provide either Lame parameters or valid Young's modulus and Poisson's ratio!"
        # elif wp.abs(lame_mu) == 0.0 and wp.abs(lame_lambda) == 0.0:
        #     lame_mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
        #     lame_lambda = (youngs_modulus * poissons_ratio) / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
        self.lame_mus = wp.zeros(youngs_modulus.shape[0], dtype=wp.float64, device=device)
        self.lame_lambdas = wp.zeros(youngs_modulus.shape[0], dtype=wp.float64, device=device)
        wp.launch(
            convert_youngs_poisson_to_lame,
            dim=youngs_modulus.shape[0],
            inputs=[youngs_modulus, poissons_ratio],
            outputs=[self.lame_mus, self.lame_lambdas],
            device=device
        )

        ### Compute inverted undeformed/reference shape matrix, mass, volume per element
        n_vertices = initial_positions.shape[0]
        n_elements = elements.shape[0]
        self.inv_Dm = wp.zeros(n_elements, dtype=wp.mat33d, device=device)
        self.masses = wp.zeros(n_elements, dtype=wp.float64, device=device)
        self.volumes = wp.zeros(n_elements, dtype=wp.float64, device=device)
        wp.launch(
            compute_element_invDm_masses_volume,
            dim=n_elements,
            inputs=[initial_positions, elements, densities],
            outputs=[self.inv_Dm, self.masses, self.volumes],
            device=device
        )

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

        self.dDs_dx = wp.array(dDs_dx, dtype=wp.mat33d, device=device)


        ### Perform graph coloring
        start_time = time.time()
        adjacency, vertex_valence = compute_adjacency_dict(elements.numpy())
        _, color_groups = graph_coloring(adjacency)
        end_time = time.time()
        print(f"Assigned {len(color_groups)} colors in {(end_time - start_time)*1000:.4f}ms. Vertex valence: {vertex_valence}")
        # Convert color groups to full array
        n_colors = len(color_groups)
        max_group_size = max(len(g) for g in color_groups.values())
        colors = np.full([n_colors, max_group_size], -1, dtype=int)
        for c in range(n_colors):
            group = color_groups[c]
            colors[c, :len(group)] = group
        self.color_groups = wp.array(colors, dtype=wp.int32, device=device)

        ### Find an adjacency mapping from vertex -> neighboring elements
        adj_v2e_list = [[] for _ in range(n_vertices)]
        max_incident_elements = 0
        for i, ele in enumerate(elements.numpy()):
            for vertex in ele:
                adj_v2e_list[vertex].append(i)
                max_incident_elements = max(max_incident_elements, len(adj_v2e_list[vertex]))
        print(f"Max incident elements per vertex: {max_incident_elements}")
        # Create fixed-size adjacency array
        adj_v2e = np.full((n_vertices, max_incident_elements), -1, dtype=int)
        for vertex in range(n_vertices):
            for j, ele_idx in enumerate(adj_v2e_list[vertex]):
                adj_v2e[vertex, j] = ele_idx
        self.adj_v2e = wp.array(adj_v2e, dtype=wp.int32, device=device)


    def reset (self) -> None:
        """
        Reset the solver state to the initial conditions.
        """
        self.old_positions = wp.clone(self.initial_positions)
        self.old_velocities = wp.zeros_like(self.initial_positions)


    def step (
        self, 
        positions: wp.array(dtype=wp.vec3d),
        dt: wp.float64,
        dx_tol: float = None
    ) -> wp.array(dtype=wp.vec3d):
        """
        Perform a single time step of the VBD solver using graph coloring for parallelization.

        Args:
            positions: Current vertex positions.
            dt: Time step size.

        Returns:
            Updated vertex positions after the time step. Does not overwrite the input positions.
        """
        n_vertices = positions.shape[0]
        n_colors = self.color_groups.shape[0]
        n_vertices_per_color = self.color_groups.shape[1]

        if dx_tol is None:
            dx_tol = self.dx_tol

        # Initial guess: explicit Euler
        self.old_positions = positions
        new_positions = wp.clone(positions)
        wp.launch(
            position_initialization,
            dim=n_vertices,
            inputs=[self.old_positions, self.old_velocities, self.gravity, dt, self.active_mask],
            outputs=[new_positions]
        )

        dxs = wp.zeros((n_vertices, 3), dtype=wp.float64)
        gradients = wp.zeros((n_vertices, self.adj_v2e.shape[1], 3), dtype=wp.float64)
        hessians = wp.zeros((n_vertices, self.adj_v2e.shape[1], 3, 3), dtype=wp.float64)

        hist = {"dx": [], "grad": []}
        for i in range(self.max_iter):
            # Loop over colors
            for c in range(n_colors):
                wp.synchronize_device(self.device)
                wp.launch(
                    accumulate_grad_hess,
                    dim=[n_vertices_per_color],
                    inputs=[
                        new_positions, self.old_positions, self.old_velocities, 
                        self.inv_Dm, self.dDs_dx, 
                        self.masses, self.volumes, self.lame_mus, self.lame_lambdas, self.damping_coefficient,
                        self.gravity, dt, 
                        self.elements, self.adj_v2e, self.color_groups[c], self.active_mask
                    ],
                    outputs=[gradients, hessians],
                    device=self.device
                )
                wp.launch_tiled(
                    solve_grad_hess,
                    dim=n_vertices,
                    block_dim=64,
                    inputs=[gradients, hessians, self.active_mask, self.color_groups[c]],
                    outputs=[dxs]
                )
                wp.launch(
                    add_dx,
                    dim=n_vertices,
                    inputs=[new_positions, dxs, self.active_mask, self.color_groups[c]],
                    outputs=[new_positions]
                )

            hist["dx"].append(abs(dxs.numpy()).mean())
            # hist["grad"].append(abs(grads.numpy()).mean())
            if abs(dxs.numpy().sum(1)).max() < dx_tol:
                break
        
        if i == self.max_iter - 1:
            print(f"Warning: VBD solver did not converge within the maximum number of iterations. Final dx max: {abs(dxs.numpy()).max()}")
        # print(f"Iteration {i}: Maximum grad: {abs(grads.numpy()).max():.2e} \tMaximum dx: {abs(dxs.numpy()).max():.2e}")

        plot = False
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(6,2))
            plt.subplots_adjust(wspace=0.4)

            axs[0].plot(np.array(hist["dx"]))
            axs[0].set_xlabel("Iteration (-)")
            axs[0].set_ylabel("Mean dx (m)")
            axs[0].grid()
            axs[0].set_xlim(0, len(hist["dx"]))
            axs[0].set_yscale("log")

            axs[1].plot(np.array(hist["grad"]))
            axs[1].set_xlabel("Iteration (-)")
            axs[1].set_ylabel("Mean grad norm (N)")
            axs[1].grid()
            axs[1].set_xlim(0, len(hist["grad"]))
            axs[1].set_yscale("log")

            fig.savefig("outputs/vbd_convergence.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


        # Discretize velocities, implicit Euler
        self.old_velocities = (new_positions - self.old_positions) / dt
        self.old_positions = new_positions

        return new_positions

