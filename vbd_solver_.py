### VBD solver implementation

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
diljk = wp.array(diljk.reshape(9,9), dtype=wp.float64)


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

    # if vertex_idx == 1:
    #     wp.printf("Vertex %d, Element %d: mass: %e\n", vertex_idx, ele_idx, masses[ele_idx])

    return gradient, hessian


@wp.func
def elastic_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    volumes: wp.array(dtype=wp.float64),
    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),
    lame_mu: wp.float64,
    lame_lambda: wp.float64,
    elements: wp.array(dtype=wp.vec4i)
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
    x = positions[vertex_idx]
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
    volume = volumes[ele_idx]

    # if vertex_idx == 1:
    #     wp.printf("Vertex %d, Element %d: volume: %e, F = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, volume, F[0,0], F[0,1], F[0,2], F[1,0], F[1,1], F[1,2], F[2,0], F[2,1], F[2,2])

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

    # Elastic, stable Neo-hookean
    mu = wp.float64(4.0) * lame_mu / wp.float64(3.0)                    # Adjusted mu for stable Neo-Hookean
    lmbda = lame_lambda + wp.float64(5.0) * lame_mu / wp.float64(6.0)   # Adjusted lambda for stable Neo-Hookean
    alpha = wp.float64(1.0) + wp.float64(0.75) * mu / lmbda

    J = wp.determinant(F)
    Ic = wp.trace(F * wp.transpose(F))
    Ft = wp.transpose(F)
    Finv = wp.inverse(F)
    FinvT = wp.transpose(Finv)
    dPhi_dF = (
        mu * F * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) 
        + lmbda * (J - alpha) * J * FinvT
    )

    # Sum up all contributions
    for i in range(3):
        for j in range(3):
            gradient[0] += volume * dPhi_dF[i, j] * dF_dx[0][i, j]
            gradient[1] += volume * dPhi_dF[i, j] * dF_dx[1][i, j]
            gradient[2] += volume * dPhi_dF[i, j] * dF_dx[2][i, j]

            hessian[i, j] += volume * (
                wp.float64(2.0) * mu / ((Ic + wp.float64(1.0))*(Ic + wp.float64(1.0))) * wp.trace(Ft * dF_dx[i]) * wp.trace(Ft * dF_dx[j])
                + mu * (wp.float64(1.0) - wp.float64(1.0) / (Ic + wp.float64(1.0))) * wp.trace(wp.transpose(dF_dx[i]) * dF_dx[j])
                + lmbda * J * (wp.float64(2.0) * J - alpha) * wp.trace(Finv * dF_dx[i]) * wp.trace(Finv * dF_dx[j])
                - lmbda * J * (J - alpha) * wp.trace(Finv * dF_dx[j] * Finv * dF_dx[i])
            )

    return gradient, hessian


@wp.func
def _elastic_gradient_hessian (
    vertex_idx: wp.int32,
    ele_idx: wp.int32,
    positions: wp.array(dtype=wp.vec3d),
    volumes: wp.array(dtype=wp.float64),
    inv_Dm: wp.array(dtype=wp.mat33d),
    dDs_dx: wp.array2d(dtype=wp.mat33d),
    lame_mu: wp.float64,
    lame_lambda: wp.float64,
    elements: wp.array(dtype=wp.vec4i)
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
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
    volume = volumes[ele_idx]

    ### Create derivatives of Ds after positions (constants)
    dF_dx = wp.zeros(shape=(3,), dtype=wp.mat33d) # shape should technically be 9x3, but we can store as 3 mat33d for simplicity
    # Trick with masks to avoid branching
    for i in range(4):
        # Figure out which vertex we are differentiating with respect to
        # mask = wp.float64(vertex_idx == ele[i])
        if vertex_idx == ele[i]:
            # Mask will only be true for one of the four vertices, sum up contributions in x, y, z
            dF_dx[0] = dDs_dx[i][0] * inv_D
            dF_dx[1] = dDs_dx[i][1] * inv_D
            dF_dx[2] = dDs_dx[i][2] * inv_D

    ### Assemble gradient and hessian
    gradient = wp.vec3d()
    hessian = wp.mat33d()

    # Elastic, St. Venant-Kirchhoff
    Ft = wp.transpose(F)
    E = wp.float64(0.5) * (Ft * F - wp.identity(3, dtype=wp.float64))
    Et = wp.transpose(E)
    trE = wp.trace(E)

    # if vertex_idx == 1 and ele_idx == 1:
    #     wp.printf("Vertex %d, Element %d in element [%d %d %d %d]\n", vertex_idx, ele_idx, ele[0], ele[1], ele[2], ele[3])
    #     wp.printf("Vertex %d, Element %d: Ds = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, Ds[0,0], Ds[0,1], Ds[0,2], Ds[1,0], Ds[1,1], Ds[1,2], Ds[2,0], Ds[2,1], Ds[2,2])
    #     wp.printf("Vertex %d, Element %d: F = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, F[0,0], F[0,1], F[0,2], F[1,0], F[1,1], F[1,2], F[2,0], F[2,1], F[2,2])
    #     wp.printf("Vertex %d, Element %d: dF_dx[0] = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, dF_dx[0][0,0], dF_dx[0][0,1], dF_dx[0][0,2], dF_dx[0][1,0], dF_dx[0][1,1], dF_dx[0][1,2], dF_dx[0][2,0], dF_dx[0][2,1], dF_dx[0][2,2])
    #     wp.printf("Vertex %d, Element %d: dF_dx[1] = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, dF_dx[1][0,0], dF_dx[1][0,1], dF_dx[1][0,2], dF_dx[1][1,0], dF_dx[1][1,1], dF_dx[1][1,2], dF_dx[1][2,0], dF_dx[1][2,1], dF_dx[1][2,2])
    #     wp.printf("Vertex %d, Element %d: dF_dx[2] = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, dF_dx[2][0,0], dF_dx[2][0,1], dF_dx[2][0,2], dF_dx[2][1,0], dF_dx[2][1,1], dF_dx[2][1,2], dF_dx[2][2,0], dF_dx[2][2,1], dF_dx[2][2,2])
    #     wp.printf("Vertex %d, Element %d: trace(E) = %e \t E = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, wp.trace(E), E[0,0], E[0,1], E[0,2], E[1,0], E[1,1], E[1,2], E[2,0], E[2,1], E[2,2])


    # Sum up all contributions
    for i in range(3):
        dEdxi = wp.float64(0.5) * (Ft * dF_dx[i] + wp.transpose(dF_dx[i]) * F)

        gradient[i] = volume * (
            lame_lambda * wp.trace(E) * wp.trace(dEdxi)
            + wp.float64(2.0) * lame_mu * wp.trace(wp.transpose(dEdxi) * E)
        )
        # gradient[i] = volume * wp.trace(
        #     wp.transpose(wp.float64(0.5) * lame_lambda * trE * (F + Ft)
        #     + lame_mu * (F * Et + Ft * E)) * dF_dx[i]
        # )

        # if vertex_idx == 1 and ele_idx == 1:
        #     wp.printf("Vertex %d, Element %d: gradient[%d] = %e \t dE_dx[%d] = [[%e %e %e], [%e %e %e], [%e %e %e]]\n", vertex_idx, ele_idx, i, gradient[i], i, dEdxi[0,0], dEdxi[0,1], dEdxi[0,2], dEdxi[1,0], dEdxi[1,1], dEdxi[1,2], dEdxi[2,0], dEdxi[2,1], dEdxi[2,2])
        #     wp.printf("Vertex %d, Element %d, index %d: volume = %e, term1 = %e, term2 = %e\n", vertex_idx, ele_idx, i, volume, lame_lambda * wp.trace(E) * wp.trace(dEdxi), wp.float64(2.0) * lame_mu * wp.trace(wp.transpose(dEdxi) * E))


        for j in range(3):
            dEdxj = wp.float64(0.5) * (Ft * dF_dx[j] + wp.transpose(dF_dx[j]) * F)
            d2E_dxidxj = wp.float64(0.5) * (wp.transpose(dF_dx[i]) * dF_dx[j] + wp.transpose(dF_dx[j]) * dF_dx[i])
            
            hessian[i, j] += volume * (
                lame_lambda * (wp.trace(dEdxi) * wp.trace(dEdxj) + wp.trace(E) * wp.trace(d2E_dxidxj))
                + wp.float64(2.0) * lame_mu * (wp.trace(wp.transpose(dEdxi) * dEdxj) + wp.trace(wp.transpose(d2E_dxidxj) * E))
            )

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
    lame_mu: wp.float64,
    lame_lambda: wp.float64,
    damping_coefficient: wp.float64,

    gravity: wp.vec3d,
    dt: wp.float64,

    elements: wp.array(dtype=wp.vec4i),
    adj_v2e: wp.array2d(dtype=wp.int32),
    color_groups: wp.array2d(dtype=wp.int32),
    active_mask: wp.array(dtype=wp.bool),

    gradients: wp.array3d(dtype=wp.float64),
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

        masses: Element masses.
        volumes: Element volumes.
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

    ### Compute gradients and hessians
    grad_inertial, hess_inertia = inertial_gradient_hessian(idx_v, idx_e, positions, old_positions, old_velocities, masses, gravity, dt)
    grad_elastic, hess_elastic = elastic_gradient_hessian(idx_v, idx_e, positions, volumes, inv_Dm, dDs_dx, lame_mu, lame_lambda, elements)
    # Add damping
    grad_damping = damping_coefficient * hess_elastic * (positions[idx_v] - old_positions[idx_v]) / dt
    hess_damping = damping_coefficient * hess_elastic / dt

    # if idx_v == 1:
    #     wp.printf("Vertex %d, Element %d: Inertial grad %f %f %f, Elastic grad %f %f %f, Damping grad %f %f %f\n", idx_v, idx_e, grad_inertial[0], grad_inertial[1], grad_inertial[2], grad_elastic[0], grad_elastic[1], grad_elastic[2], grad_damping[0], grad_damping[1], grad_damping[2])

    ### Accumulate results
    for k in range(3):
        gradients[idx_v, j, k] = grad_inertial[k] + grad_elastic[k] + grad_damping[k]
        for l in range(3):
            hessians[idx_v, j, k, l] = hess_inertia[k, l] + hess_elastic[k, l] + hess_damping[k, l]


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
    # wp.tile_store(dx[i], -wp.float64(0.000001)*total_grad)
    


@wp.kernel
def add_dx (
    positions: wp.array(dtype=wp.vec3d),
    dx: wp.array2d(dtype=wp.float64),

    new_positions: wp.array(dtype=wp.vec3d)
) -> None:
    i = wp.tid()

    new_positions[i] = positions[i] + wp.vec3d(dx[i,0], dx[i,1], dx[i,2])

    # for j in range(3):
    #     new_positions[i][j] = positions[i][j] + dx[i][j]


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
        new_positions[i][j] = positions[i][j] + velocities[i][j] * dt #+ gravity[j] * dt * dt


@wp.kernel
def compute_element_masses_volume (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    densities: wp.array(dtype=wp.float64),

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
    volume = wp.abs(wp.determinant(Dm)) / wp.float64(6.0)
    masses[i] = density * volume
    volumes[i] = volume

    # Distribute mass equally to the four vertices, TODO: INEFFICIENT, CHANGE TO TILES
    # for k in range(4):
    #     wp.atomic_add(masses, ele[k], mass / wp.float64(4.0))


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
    # Check if inversion was successful
    res = Dm * inv_Dm[i]
    for r in range(3):
        for c in range(3):
            if r == c:
                assert wp.abs(res[r,c] - wp.float64(1.0)) < EPS, "Singular matrix encountered in inv_Dm computation!"
            else:
                assert wp.abs(res[r,c]) < EPS, "Singular matrix encountered in inv_Dm computation!"


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
            damping_coefficient: wp.float64 = wp.float64(1.0),

            active_mask: wp.array(dtype=wp.bool)=None,
            gravity: wp.vec3d = wp.vec3d(0.0, 0.0, -9.81),

            device: str = "cpu"
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
        self.damping_coefficient = damping_coefficient
        self.gravity = gravity
        self.device = device

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


        ### Compute mass per element from element densities
        n_elements = elements.shape[0]
        self.masses = wp.zeros(n_elements, dtype=wp.float64, device=device)
        self.volumes = wp.zeros(n_elements, dtype=wp.float64, device=device)
        wp.launch(
            compute_element_masses_volume,
            dim=n_elements,
            inputs=[initial_positions, elements, densities],
            outputs=[self.masses, self.volumes],
            device=device
        )

        ### Compute inverted undeformed/reference shape matrix for tetrahedrons.
        self.inv_Dm = wp.zeros(n_elements, dtype=wp.mat33d, device=device)
        wp.launch(
            compute_inv_Dm,
            dim=n_elements,
            inputs=[initial_positions, elements],
            outputs=[self.inv_Dm],
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
        # Initial guess: explicit Euler
        new_positions = wp.clone(positions)
        wp.launch(
            position_initialization,
            dim=n_vertices,
            inputs=[self.old_positions, self.old_velocities, self.gravity, dt, self.active_mask],
            outputs=[new_positions]
        )

        # TODO: Lot of memory use, optimization possible
        gradients = wp.zeros((n_vertices, n_elements_per_vertex, 3), dtype=wp.float64, device=self.device)
        hessians = wp.zeros((n_vertices, n_elements_per_vertex, 3, 3), dtype=wp.float64, device=self.device)

        hist = {"dx": [], "grad": []}
        MAX_ITER = 100
        for i in range(MAX_ITER):
            wp.synchronize_device(self.device)
            wp.launch(
                accumulate_grad_hess,
                dim=[n_colors, n_vertices_per_color, n_elements_per_vertex],
                inputs=[
                    new_positions, self.old_positions, self.old_velocities, 
                    self.inv_Dm, self.dDs_dx, 
                    self.masses, self.volumes, self.lame_mu, self.lame_lambda, self.damping_coefficient,
                    self.gravity, dt, 
                    self.elements, self.adj_v2e, self.color_groups, self.active_mask
                ],
                outputs=[gradients, hessians],
                device=self.device
            )
            dx = wp.zeros((n_vertices, 3), dtype=wp.float64, device=self.device)
            wp.synchronize_device(self.device)
            wp.launch_tiled(
                solve_grad_hess,
                dim=n_vertices,
                block_dim=64,
                inputs=[gradients, hessians, self.active_mask],
                outputs=[dx],
                device=self.device
            )
            wp.synchronize_device(self.device)
            breakpoint()
            wp.launch(
                add_dx,
                dim=n_vertices,
                inputs=[new_positions, dx],
                outputs=[new_positions],
                device=self.device
            )
            wp.synchronize_device(self.device)
            # print(f"Iteration {i}: Maximum gradient: {abs(gradients.numpy().sum(1)).max():.4e} and \tMaximum dx: {abs(dx.numpy()).max():.2e}")
            # breakpoint()
            hist["dx"].append(abs(dx.numpy()).mean())
            hist["grad"].append(abs(gradients.numpy().sum(1)).mean())

            if abs(gradients.numpy().sum(1)).max() < 1e-6:
                break
        
        # breakpoint()
        if i == MAX_ITER - 1:
            print(f"Warning: VBD solver did not converge within the maximum number of iterations. Final dx max: {abs(dx.numpy()).max()}")

        print(f"Iteration {i}: Maximum gradient: {abs(gradients.numpy().sum(1)).max():.4e} and \tMaximum dx: {abs(dx.numpy()).max():.2e}")

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

        # breakpoint()


        # Discretize velocities, implicit Euler
        self.old_velocities = (new_positions - positions) / dt
        # Set old positions for next velocities computation
        self.old_positions = new_positions

        # Check if nan because hessian singlar TODO
        # print(f"All not nan: {wp.isnan(new_positions)}")

        return new_positions

