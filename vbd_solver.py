### VBD solver implementation

import numpy as np
import warp as wp

EPS = 1e-12
MAX_ELEMENTS_PER_VERTEX = 128

DELTA = 1e-6

@wp.func
def compute_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    old_positions: wp.array(dtype=wp.vec3d),
    old_velocities: wp.array(dtype=wp.vec3d),
    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),
    masses: wp.array(dtype=wp.float64),
    lame_mu: wp.float64,
    lame_lambda: wp.float64,
    gravity: wp.vec3d,
    dt: wp.float64,
    elements: wp.array(dtype=wp.vec4i)
# ) -> tuple[wp.vec3d, wp.mat33d]:
):
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
    x = positions[vertex_idx]
    x_other = x + wp.vec3d(wp.float64(DELTA), wp.float64(0.0), wp.float64(0.0))  # For finite difference check
    x_prev = old_positions[vertex_idx]
    v_prev = old_velocities[vertex_idx]
    m = masses[vertex_idx]
    ele = elements[ele_idx]
    inv_D = inv_Dm[ele_idx]

    ### Compute deformation gradient F
    x0 = positions[ele[0]]
    x1 = positions[ele[1]]
    x2 = positions[ele[2]]
    x3 = positions[ele[3]]

    x0_other = x_other if vertex_idx == ele[0] else positions[ele[0]]
    x1_other = x_other if vertex_idx == ele[1] else positions[ele[1]]
    x2_other = x_other if vertex_idx == ele[2] else positions[ele[2]]
    x3_other = x_other if vertex_idx == ele[3] else positions[ele[3]]

    # Deformed shape matrix # TODO write out matmul and skip mat33d initialization for performance
    Ds = wp.matrix_from_cols(
        x0 - x3,
        x1 - x3,
        x2 - x3
    )
    Ds_other = wp.matrix_from_cols(
        x0_other - x3_other,
        x1_other - x3_other,
        x2_other - x3_other
    )
    F = Ds * inv_D
    F_other = Ds_other * inv_D
    volume = wp.float64(1.0) / (wp.abs(wp.determinant(inv_D)) * wp.float64(6.0)) # TODO: could be optimized, or we precompute.

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
    gradient_other = wp.vec3d()  # For finite difference check
    hessian = wp.mat33d()

    # Kinetic
    y = x_prev + dt * v_prev # + external accelerations, if any. TODO
    gradient += m  / (dt * dt) * (x - y)
    gradient_other += m  / (dt * dt) * (x_other - y)
    hessian += m / (dt * dt) * wp.identity(3, dtype=wp.float64)

    # Gravity
    gradient += -m * gravity
    gradient_other += -m * gravity
    # no hessian

    # Elastic, stable Neo-hookean
    mu = wp.float64(4.0) * lame_mu / wp.float64(3.0)                    # Adjusted mu for stable Neo-Hookean
    lmbda = lame_lambda + wp.float64(5.0) * lame_mu / wp.float64(6.0)   # Adjusted lambda for stable Neo-Hookean
    alpha = wp.float64(1.0) + wp.float64(0.75) * mu / lmbda

    J = wp.determinant(F)
    J_other = wp.determinant(F_other)
    Ic = wp.trace(F * wp.transpose(F))
    Ic_other = wp.trace(F_other * wp.transpose(F_other))
    Ft = wp.transpose(F)
    Finv = wp.inverse(F)
    Finv_other = wp.inverse(F_other)
    FinvT = wp.transpose(Finv)
    FinvT_other = wp.transpose(Finv_other)
    dPhi_dF = (
        mu * F * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) 
        + lmbda * (J - alpha) * J * FinvT
    )
    dPhi_dF_other = (
        mu * F_other * (wp.float64(1.0) - wp.float64(1.0) / (Ic_other + wp.float64(1.0))) 
        + lmbda * (J_other - alpha) * J_other * FinvT_other
    )
        

    # Hessian has a complex form, we break it down to dF^T A dF, where A is 9 separate 3x3 blocks being the second derivative dPhi_dF2, each having 4 terms
    # dPhi_dF2 = wp.zeros((9,), dtype=wp.mat33d)
    # for i in range(3):
    #     for j in range(3):
    #         mask = wp.float64(i == j)
    #         dPhi_dF2[3*i+j] = (
    #             mask * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) * mu * wp.identity(3, dtype=wp.float64)
    #             # + (wp.float64(2.0) * J - alpha) * lmbda * J * (wp.outer(FinvT[i], FinvT[j]))
    #             # - lmbda * (J - alpha) * J * (wp.outer(FinvT[j], FinvT[i]))
    #             + wp.float64(2.0) * mu / ((Ic + wp.float64(1.0))*(Ic + wp.float64(1.0))) * wp.outer(F[i], F[j])
    #         )

    # Sum up all contributions
    for i in range(3):
        for j in range(3):
            gradient[0] += volume * dPhi_dF[i, j] * dF_dx[0][i, j]
            gradient[1] += volume * dPhi_dF[i, j] * dF_dx[1][i, j]
            gradient[2] += volume * dPhi_dF[i, j] * dF_dx[2][i, j]

            gradient_other[0] += volume * dPhi_dF_other[i, j] * dF_dx[0][i, j]
            gradient_other[1] += volume * dPhi_dF_other[i, j] * dF_dx[1][i, j]
            gradient_other[2] += volume * dPhi_dF_other[i, j] * dF_dx[2][i, j]

            hessian[i, j] += volume * (
                wp.float64(2.0) * mu / ((Ic + wp.float64(1.0))*(Ic + wp.float64(1.0))) * wp.trace(Ft * dF_dx[i]) * wp.trace(Ft * dF_dx[j])
                + mu * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) * wp.trace(wp.transpose(dF_dx[i]) * dF_dx[j])
                + lmbda * J * (wp.float64(2.0) * J - alpha) * wp.trace(Finv * dF_dx[i]) * wp.trace(Finv * dF_dx[j])
                - lmbda * J * (J - alpha) * wp.trace(Finv * dF_dx[j] * Finv * dF_dx[i])
            )

    return gradient, gradient_other, hessian


@wp.kernel
def accumulate_grad_hess (
    positions: wp.array(dtype=wp.vec3d),
    old_positions: wp.array(dtype=wp.vec3d),
    old_velocities: wp.array(dtype=wp.vec3d),

    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),

    masses: wp.array(dtype=wp.float64),
    lame_mu: wp.float64,
    lame_lambda: wp.float64,

    gravity: wp.vec3d,
    dt: wp.float64,

    elements: wp.array(dtype=wp.vec4i),
    adj_v2e: wp.array2d(dtype=wp.int32),
    color_groups: wp.array2d(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.bool),

    gradients: wp.array3d(dtype=wp.float64),
    gradients_other: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64)
) -> None:
    """
    Accumulate gradients and Hessians for each vertex-element pair in parallel using graph coloring.

    Inputs:
        positions: Current vertex positions.
        old_positions: Previous vertex positions.
        old_velocities: Previous vertex velocities.

        inv_Dm: Inverted undeformed shape matrices for elements.
        dDs_dx: Derivatives of Ds with respect to positions. Shape [4, 3, 3, 3], a 3x3 matrix for each vertex of the tetrahedron.

        masses: Vertex masses.
        lame_mu: Lame parameter mu.
        lame_lambda: Lame parameter lambda.

        gravity: Gravity vector.
        dt: Time step size.

        elements: Mesh elements.
        adj_v2e: Adjacency mapping from vertex to neighboring elements.
        color_groups: Color assignments for vertices.
        active_mask: Mask indicating active vertices.
    
    Outputs:
        gradients: Output array for gradients of every element of each vertex.
        hessians: Output array for Hessians of every element of each vertex.
    """
    c, i, j = wp.tid()

    # Skip if this is a padding entry
    idx_v = color_groups[c, i]
    if idx_v == -1:
        return
    idx_e = adj_v2e[idx_v, j]
    if idx_e == -1:
        return
    
    # Skip if vertex is not active
    if not active_mask[idx_v]:
        return

    grad, grad_other, hess = compute_gradient_hessian(idx_v, idx_e, positions, old_positions, old_velocities, inv_Dm, dDs_dx, masses, lame_mu, lame_lambda, gravity, dt, elements)
    for k in range(3):
        gradients[idx_v, j, k] = grad[k]
        gradients_other[idx_v, j, k] = grad_other[k]
        for l in range(3):
            hessians[idx_v, j, k, l] = hess[k, l]



@wp.kernel
def solve_grad_hess (
    gradients: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64),
    active_mask: wp.array(dtype=wp.bool),

    dx: wp.array2d(dtype=wp.float64)
) -> None:
    i = wp.tid()

    # Skip if vertex is not active
    if not active_mask[i]:
        return
    
    grad = wp.tile_load(gradients[i], (MAX_ELEMENTS_PER_VERTEX, 3))
    hess = wp.tile_load(hessians[i], (MAX_ELEMENTS_PER_VERTEX, 3, 3))
    # Sum up contributions of elements to vertices
    total_grad = wp.tile_sum(grad, axis=0)
    total_hess = wp.tile_sum(hess, axis=0)

    # Solve for dx
    L = wp.tile_cholesky(total_hess)
    out = wp.tile_cholesky_solve(L, -total_grad)
    
    # Store results
    wp.tile_store(dx[i], out)
    


@wp.kernel
def add_dx (
    positions: wp.array(dtype=wp.vec3d),
    dx: wp.array2d(dtype=wp.float64),

    new_positions: wp.array(dtype=wp.vec3d)
) -> None:
    i = wp.tid()

    for j in range(3):
        new_positions[i][j] = positions[i][j] + dx[i][j]


@wp.kernel
def compute_vertex_masses (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    densities: wp.array(dtype=wp.float64),

    masses: wp.array(dtype=wp.float64)
) -> None:
    """
    Compute mass per vertex from element densities.

    Args:
        positions: Vertex positions.
        elements: Mesh elements.
        densities: Element densities.
        masses: Output array for vertex masses.
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
    volume = wp.abs(wp.determinant(Dm)) / wp.float64(6.0)
    mass = density * volume

    # Distribute mass equally to the four vertices, TODO: INEFFICIENT, CHANGE TO TILES
    for k in range(4):
        wp.atomic_add(masses, ele[k], mass / wp.float64(4.0))


@wp.kernel
def compute_inv_Dm (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),

    inv_Dm: wp.array(dtype=wp.mat33d)
) -> None:
    """
    Compute the inverted undeformed/reference shape matrix for each tetrahedral element.

    Args:
        positions: Vertex positions.
        elements: Mesh elements.
        inv_Dm: Output array for inverted shape matrices.
    """
    i = wp.tid()

    ele = elements[i]
    X0 = positions[ele[0]]
    X1 = positions[ele[1]]
    X2 = positions[ele[2]]
    X3 = positions[ele[3]]

    # Each column is an edge vector
    Dm = wp.matrix_from_cols(
        X0 - X3,
        X1 - X3,
        X2 - X3
    )
    inv_Dm[i] = wp.inverse(Dm)


class VBDSolver:
    def __init__(
            self, 
            initial_positions: wp.array(dtype=wp.vec3d),
            elements: wp.array(dtype=wp.vec4i),
            adj_v2e: wp.array(dtype=wp.int32),
            color_groups: wp.array(dtype=wp.int32),

            densities: wp.array(dtype=wp.float64),
            lame_mu: wp.float64 = wp.float64(0.0),
            lame_lambda: wp.float64 = wp.float64(0.0),
            youngs_modulus: wp.float64 = wp.float64(0.0),
            poisson_ratio: wp.float64 = wp.float64(0.5),

            active_mask: wp.array(dtype=wp.bool)=None,
            gravity: wp.vec3d = wp.vec3d(0.0, 0.0, -9.81),
        ) -> None:
        """
        Initialize the VBD solver with the tetrahedral mesh and simulation parameters.
        Args:
            initial_positions: Initial vertex positions.
            elements: Mesh elements.
            adj_v2e: Adjacency mapping from vertex to neighboring elements.
            color_groups: Color assignments for vertices.

            densities: Element densities.
            lame_mu: Lame parameter mu.
            lame_lambda: Lame parameter lambda.
            youngs_modulus: Young's modulus (not used if Lame parameters are provided).
            poisson_ratio: Poisson's ratio (not used if Lame parameters are provided).

            gravity: Gravity vector.
            active_mask: Optional mask to indicate active vertices.
        """
        self.initial_positions = initial_positions
        self.old_positions = wp.clone(initial_positions) # For kinetic energy
        self.old_velocities = wp.zeros_like(initial_positions)
        self.elements = elements
        self.adj_v2e = adj_v2e
        self.color_groups = color_groups
        self.densities = densities
        self.gravity = gravity
        if active_mask is not None:
            self.active_mask = active_mask
        else:
            self.active_mask = wp.ones(initial_positions.shape[0], dtype=wp.bool)
    
        # If Lame parameters are not provided, compute them from Young's modulus and Poisson's ratio
        if (wp.abs(lame_mu) == 0.0 and wp.abs(lame_lambda) == 0.0) and (wp.abs(youngs_modulus) == 0.0 or wp.abs(poisson_ratio) >= 0.5):
            assert False, "Provide either Lame parameters or valid Young's modulus and Poisson's ratio!"
        elif wp.abs(lame_mu) == 0.0 and wp.abs(lame_lambda) == 0.0:
            lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_lambda = (youngs_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda


        ### Compute mass per vertex from element densities
        n_elements = elements.shape[0]
        self.masses = wp.zeros(initial_positions.shape[0], dtype=wp.float64)
        wp.launch(
            compute_vertex_masses,
            dim=n_elements,
            inputs=[initial_positions, elements, densities],
            outputs=[self.masses]
        )

        ### Compute inverted undeformed/reference shape matrix for tetrahedrons.
        self.inv_Dm = wp.zeros(n_elements, dtype=wp.mat33d)
        wp.launch(
            compute_inv_Dm,
            dim=n_elements,
            inputs=[initial_positions, elements],
            outputs=[self.inv_Dm]
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

        self.dDs_dx = wp.array(dDs_dx, dtype=wp.mat33d)


    def step (
        self, 
        positions: wp.array(dtype=wp.vec3d),
        dt: wp.float64
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
        n_elements_per_vertex = self.adj_v2e.shape[1]
        new_positions = wp.clone(positions)

        # TODO: Lot of memory use, optimization possible
        gradients = wp.zeros((n_vertices, n_elements_per_vertex, 3), dtype=wp.float64)
        gradients_other = wp.zeros((n_vertices, n_elements_per_vertex, 3), dtype=wp.float64)
        hessians = wp.zeros((n_vertices, n_elements_per_vertex, 3, 3), dtype=wp.float64)

        MAX_ITER = 100
        for i in range(MAX_ITER):
            wp.launch(
                accumulate_grad_hess,
                dim=[n_colors, n_vertices_per_color, n_elements_per_vertex],
                inputs=[new_positions, self.old_positions, self.old_velocities, self.inv_Dm, self.dDs_dx, self.masses, self.lame_mu, self.lame_lambda, self.gravity, dt, self.elements, self.adj_v2e, self.color_groups, self.active_mask],
                outputs=[gradients, gradients_other, hessians]
            )
            dx = wp.zeros((n_vertices, 3), dtype=wp.float64)
            wp.launch_tiled(
                solve_grad_hess,
                dim=n_vertices,
                block_dim=64,
                inputs=[gradients, hessians, self.active_mask],
                outputs=[dx]
            )
            wp.launch(
                add_dx,
                dim=n_vertices,
                inputs=[new_positions, dx],
                outputs=[new_positions]
            )
            # print(abs(dx.numpy()).max())
            # breakpoint()
            if abs(dx.numpy()).max() < 1e-9:
                break
        
        if i == MAX_ITER - 1:
            print("Warning: VBD solver did not converge within the maximum number of iterations.")

        # print(f"Final dx in {i} iterations: {abs(dx.numpy()).max()}")

        # Discretize velocities, implicit Euler
        self.old_velocities = (new_positions - positions) / dt
        # Set old positions for next velocities computation
        self.old_positions = new_positions

        # Check if nan because hessian singlar TODO
        # print(f"All not nan: {wp.isnan(new_positions)}")

        return new_positions

