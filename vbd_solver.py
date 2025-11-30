### VBD solver implementation

import numpy as np
import warp as wp

EPS = 1e-12
MAX_ELEMENTS_PER_VERTEX = 128

@wp.func
def compute_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    velocities: wp.array(dtype=wp.vec3d),
    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),
    masses: wp.array(dtype=wp.float64),
    lame_mu: wp.float64,
    lame_lambda: wp.float64,
    gravity: wp.vec3d,
    dt: wp.float64,
    elements: wp.array(dtype=wp.vec4i)
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
    x = positions[vertex_idx]
    v = velocities[vertex_idx]
    m = masses[ele_idx] * wp.float64(0.25) # Each vertex gets one-fourth of the element mass
    ele = elements[ele_idx]
    inv_D = inv_Dm[ele_idx]

    ### Compute deformation gradient F
    x0 = positions[ele[0]]
    x1 = positions[ele[1]]
    x2 = positions[ele[2]]
    x3 = positions[ele[3]]

    # Deformed shape matrix # TODO write out matmul and skip mat33d initialization for performance
    Ds = wp.matrix_from_cols(
        x0 - x3,
        x1 - x3,
        x2 - x3
    )
    F = Ds * inv_D
    volume = wp.abs(wp.determinant(Ds)) / wp.float64(6.0) # TODO: could be optimized, or we precompute.

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

    # Kinetic
    y = x + dt * v # + external accelerations, if any. TODO
    gradient += m  / (dt * dt) * (x - y)
    hessian += m / (dt * dt) * wp.identity(3, dtype=wp.float64)

    # Gravity
    gradient += -m * gravity
    # no hessian

    # Elastic, stable Neo-hookean
    mu = wp.float64(4.0) * lame_mu / wp.float64(3.0)                    # Adjusted mu for stable Neo-Hookean
    lmbda = lame_lambda + wp.float64(5.0) * lame_mu / wp.float64(6.0)   # Adjusted lambda for stable Neo-Hookean
    alpha = wp.float64(1.0) + wp.float64(0.75) * lame_mu / lame_lambda

    J = wp.determinant(F)
    Ic = wp.trace(F * wp.transpose(F))
    dPhi_dF = (mu * F * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) + lmbda * (J - alpha) * J * wp.transpose(wp.inverse(F)))

    # Sum up all contributions
    for i in range(3):
        for j in range(3):
            gradient[0] += volume * dPhi_dF[i, j] * dF_dx[0][i, j]
            gradient[1] += volume * dPhi_dF[i, j] * dF_dx[1][i, j]
            gradient[2] += volume * dPhi_dF[i, j] * dF_dx[2][i, j]
    

    return gradient, hessian


@wp.kernel
def accumulate_grad_hess (
    positions: wp.array(dtype=wp.vec3d),
    velocities: wp.array(dtype=wp.vec3d),
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

    gradients: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64)
) -> None:
    """
    Accumulate gradients and Hessians for each vertex-element pair in parallel using graph coloring.

    Inputs:
        positions: Current vertex positions.
        velocities: Current vertex velocities.
        inv_Dm: Inverted undeformed shape matrices for elements.
        dDs_dx: Derivatives of Ds with respect to positions. Shape [4, 3, 3, 3], a 3x3 matrix for each vertex of the tetrahedron.
        masses: Element masses.
        lame_mu: Lame parameter mu.
        lame_lambda: Lame parameter lambda.
        gravity: Gravity vector.
        dt: Time step size.
        elements: Mesh elements.
        adj_v2e: Adjacency mapping from vertex to neighboring elements.
        color_groups: Color assignments for vertices.
    
    Outputs:
        gradients: Output array for gradients of every element of each vertex.
        hessians: Output array for Hessians of every element of each vertex.
    """
    c, i, j = wp.tid()

    idx_v = color_groups[c, i]
    idx_e = adj_v2e[idx_v, j]
    
    # Skip if this is a padding entry
    if idx_v == -1 or idx_e == -1:
        return
    
    grad, hess = compute_gradient_hessian(idx_v, idx_e, positions, velocities, inv_Dm, dDs_dx, masses, lame_mu, lame_lambda, gravity, dt, elements)

    for k in range(3):
        gradients[idx_v, j, k] = grad[k]
        for l in range(3):
            hessians[idx_v, j, k, l] = hess[k, l]


@wp.kernel
def solve_grad_hess (
    gradients: wp.array3d(dtype=wp.float64),
    hessians: wp.array4d(dtype=wp.float64),

    dx: wp.array2d(dtype=wp.float64),
) -> None:
    i = wp.tid()

    grad = wp.tile_load(gradients[i], (MAX_ELEMENTS_PER_VERTEX, 3))
    hess = wp.tile_load(hessians[i], (MAX_ELEMENTS_PER_VERTEX, 3, 3))
    # Sum up contributions of elements to vertices
    total_grad = wp.tile_sum(grad, axis=0)
    total_hess = wp.tile_sum(hess, axis=0)

    # Solve for dx
    L = wp.tile_cholesky(total_hess)
    res = wp.tile_cholesky_solve(L, -total_grad)
    wp.tile_store(dx[i], res)
    


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

            masses: wp.array(dtype=wp.float64),
            lame_mu: wp.float64 = wp.float64(0.0),
            lame_lambda: wp.float64 = wp.float64(0.0),
            youngs_modulus: wp.float64 = wp.float64(0.0),
            poisson_ratio: wp.float64 = wp.float64(0.5),

            gravity: wp.vec3d = wp.vec3d(0.0, 0.0, -9.81),
        ) -> None:
        """
        Initialize the VBD solver with the tetrahedral mesh and simulation parameters.
        Args:
            initial_positions: Initial vertex positions.
            elements: Mesh elements.
            adj_v2e: Adjacency mapping from vertex to neighboring elements.
            color_groups: Color assignments for vertices.
            masses: Mass if each tetrahedral element.
            lame_mu: Lame parameter mu.
            lame_lambda: Lame parameter lambda.
            youngs_modulus: Young's modulus (not used if Lame parameters are provided).
            poisson_ratio: Poisson's ratio (not used if Lame parameters are provided).
            gravity: Gravity vector.
        """
        self.initial_positions = initial_positions
        self.old_positions = wp.clone(initial_positions) # For velocities computation
        self.elements = elements
        self.adj_v2e = adj_v2e
        self.color_groups = color_groups
        self.masses = masses
        self.gravity = gravity
    
        # If Lame parameters are not provided, compute them from Young's modulus and Poisson's ratio
        if (lame_mu == 0.0 and lame_lambda == 0.0) and (youngs_modulus != 0.0 and poisson_ratio < 0.5):
            lame_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
            lame_lambda = (youngs_modulus * poisson_ratio) / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda

        ### Compute inverted undeformed/reference shape matrix for tetrahedrons.
        n_elements = elements.shape[0]
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

        # Discretize velocities, implicit Euler
        velocities = (positions - self.old_positions) / dt

        # TODO: Lot of memory use, optimization possible
        gradients = wp.zeros((n_vertices, self.adj_v2e.shape[1], 3), dtype=wp.float64)
        hessians = wp.zeros((n_vertices, self.adj_v2e.shape[1], 3, 3), dtype=wp.float64)

        wp.launch(
            accumulate_grad_hess,
            dim=[self.color_groups.shape[0], self.color_groups.shape[1], self.adj_v2e.shape[1]],
            inputs=[positions, velocities, self.inv_Dm, self.dDs_dx, self.masses, self.lame_mu, self.lame_lambda, self.gravity, dt, self.elements, self.adj_v2e, self.color_groups],
            outputs=[gradients, hessians]
        )
        dx = wp.zeros((n_vertices, 3), dtype=wp.float64)
        wp.launch_tiled(
            solve_grad_hess,
            dim=n_vertices,
            block_dim=64,
            inputs=[gradients, hessians],
            outputs=[dx]
        )
        new_positions = wp.zeros_like(positions)
        wp.launch(
            add_dx,
            dim=n_vertices,
            inputs=[positions, dx],
            outputs=[new_positions]
        )
        # Set old positions for next velocities computation
        self.old_positions = positions

        # Check if nan because hessian singlar TODO
        # print(f"All not nan: {wp.isnan(new_positions)}")
        
        # print(f"Old positions: {positions.numpy()}")
        # print(f"Computed dx: {dx.numpy()}")
        # print(f"New positions: {new_positions.numpy()}")

        return new_positions

