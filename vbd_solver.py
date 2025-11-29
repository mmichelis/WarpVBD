### VBD solver implementation

import numpy as np
import warp as wp


@wp.func
def compute_gradient (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    vertex_idx: wp.int32,
    ele_idx: wp.int32
) -> wp.vec3d:
    """
    Compute the gradient of the energy with respect to vertex positions of a specific element.

    Args:
        positions: Current vertex positions.
        elements: Mesh elements.
        vertex_idx: Index of the vertex.
        ele_idx: Index of the element.

    Returns:
        Gradient array.
    """
    # Placeholder implementation
    gradient = wp.vec3d(wp.float64(1.0))
    return gradient

@wp.func
def compute_hessian (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    vertex_idx: wp.int32,
    ele_idx: wp.int32
) -> wp.mat33d:
    """
    Compute the Hessian of the energy with respect to vertex positions of a specific element.

    Args:
        positions: Current vertex positions.
        elements: Mesh elements.
        vertex_idx: Index of the vertex.
        ele_idx: Index of the element.

    Returns:
        Hessian matrix.
    """
    # Placeholder implementation
    hessian = wp.identity(3, dtype=wp.float64)
    return hessian


@wp.kernel
def accumulate_grad_hess (
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    adj_v2e: wp.array2d(dtype=wp.int32),
    color_groups: wp.array2d(dtype=wp.int32),
    gradients: wp.array(dtype=wp.float64, shape=(1, 1, 3)),
    hessians: wp.array(dtype=wp.float64, shape=(1, 1, 3, 3))
) -> None:
    """
    Accumulate gradients and Hessians for each vertex-element pair in parallel using graph coloring.

    Inputs:
        positions: Current vertex positions.
        elements: Mesh elements.
        adj_v2e: Adjacency mapping from vertex to neighboring elements.
        color_groups: Color assignments for vertices.
    
    Outputs:
        gradients: Output array for gradients.
        hessians: Output array for Hessians.
    """
    c, i, j = wp.tid()

    idx_v = color_groups[c, i]
    idx_e = adj_v2e[idx_v, j]
    
    # Skip if this is a padding entry
    if idx_v == -1 or idx_e == -1:
        return
    
    grad = compute_gradient(positions, elements, idx_v, idx_e)
    hess = compute_hessian(positions, elements, idx_v, idx_e)

    for k in range(3):
        gradients[idx_v, j, k] = grad[k]
        for l in range(3):
            hessians[idx_v, j, k, l] = hess[k, l]


@wp.kernel
def solve_grad_hess (
    gradients: wp.array(dtype=wp.float64, shape=(1, 1, 3)),
    hessians: wp.array(dtype=wp.float64, shape=(1, 1, 3, 3)),
    dx: wp.array(dtype=wp.float64, shape=(1, 3))
) -> None:
    i = wp.tid()

    grad = wp.tile_load(gradients[i], (256, 3))
    hess = wp.tile_load(hessians[i], (256, 3, 3))
    # Sum up contributions of elements to vertices
    total_grad = wp.tile_sum(grad, axis=0)
    total_hess = wp.tile_sum(hess, axis=0)

    # Solve for dx
    L = wp.tile_cholesky(total_hess)
    res = wp.tile_cholesky_solve(L, -total_grad)
    wp.tile_store(dx[i], res)
    # dx[i] += res


# @wp.kernel
# def add_dx (
#     positions: wp.array(dtype=wp.vec3d),
#     dx: wp.array(dtype=wp.vec3d)
# ) -> None:
#     i = wp.tid()
#     positions[i] = positions[i] + dx[i]


def step (    
    positions: wp.array(dtype=wp.vec3d),
    elements: wp.array(dtype=wp.vec4i),
    adj_v2e: wp.array(dtype=wp.int32),
    color_groups: wp.array(dtype=wp.int32),
    dt: float
) -> wp.array(dtype=wp.vec3d):
    """
    Perform a single time step of the VBD solver using graph coloring for parallelization.

    Args:
        positions: Current vertex positions.
        elements: Mesh elements.
        adj_v2e: Adjacency mapping from vertex to neighboring elements.
        color_groups: Color assignments for vertices.
        dt: Time step size.

    Returns:
        Updated vertex positions after the time step.
    """
    n_vertices = positions.shape[0]

    gradients = wp.zeros((n_vertices, adj_v2e.shape[1], 3), dtype=wp.float64)
    hessians = wp.zeros((n_vertices, adj_v2e.shape[1], 3, 3), dtype=wp.float64)

    wp.launch(
        accumulate_grad_hess,
        dim=[color_groups.shape[0], color_groups.shape[1], adj_v2e.shape[1]],
        inputs=[positions, elements, adj_v2e, color_groups],
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
    # new_positions = positions + dx
    breakpoint()

    return new_positions

