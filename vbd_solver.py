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
    masses: wp.array(dtype=wp.float64),
    gravity: wp.vec3d,
    dt: wp.float64,
    elements: wp.array(dtype=wp.vec4i)
) -> tuple[wp.vec3d, wp.mat33d]:
    """
    Compute the gradient and hessian of the energy with respect to vertex positions of a specific element.
    """
    x = positions[vertex_idx]
    v = velocities[vertex_idx]
    m = masses[vertex_idx]

    gradient = wp.vec3d()
    hessian = wp.mat33d()
    # Kinetic
    y = x + dt * v # + external accelerations, if any. TODO
    gradient += m  / (dt * dt) * (x - y)
    hessian += m / (dt * dt) * wp.identity(3, dtype=wp.float64)

    # Gravity
    gradient += m * gravity
    # no hessian
    
    return gradient, hessian


@wp.kernel
def accumulate_grad_hess (
    positions: wp.array(dtype=wp.vec3d),
    velocities: wp.array(dtype=wp.vec3d),
    masses: wp.array(dtype=wp.float64),
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
        masses: Vertex masses.
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
    
    grad, hess = compute_gradient_hessian(idx_v, idx_e, positions, velocities, masses, gravity, dt, elements)

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



class VBDSolver:
    def __init__(
            self, 
            initial_positions: wp.array(dtype=wp.vec3d),
            elements: wp.array(dtype=wp.vec4i),
            adj_v2e: wp.array(dtype=wp.int32),
            color_groups: wp.array(dtype=wp.int32),
            masses: wp.array(dtype=wp.float64),
            gravity: wp.vec3d = wp.vec3d(0.0, 0.0, -9.81),
        ) -> None:
        """
        Initialize the VBD solver with the mesh and simulation parameters.
        Args:
            initial_positions: Initial vertex positions.
            elements: Mesh elements.
            adj_v2e: Adjacency mapping from vertex to neighboring elements.
            color_groups: Color assignments for vertices.
            masses: Masses of the vertices.
            gravity: Gravity vector.
        """
        self.initial_positions = initial_positions
        self.old_positions = wp.clone(initial_positions) # For velocities computation
        self.elements = elements
        self.adj_v2e = adj_v2e
        self.color_groups = color_groups
        self.masses = masses
        self.gravity = gravity


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
            inputs=[positions, velocities, self.masses, self.gravity, dt, self.elements, self.adj_v2e, self.color_groups],
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
        
        # print(f"Old positions: {positions.numpy()}")
        # print(f"Computed dx: {dx.numpy()}")
        # print(f"New positions: {new_positions.numpy()}")

        return new_positions

