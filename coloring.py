### Implement graph coloring for meshes
import numpy as np
import warp as wp



# @wp.kernel
# def kernel_resolve_conflicts (
#     coloring: wp.array(dtype=wp.int32),
#     vertices_inspect: wp.array(dtype=wp.int32),
#     adjacency: wp.array(dtype=wp.int32),


def assign_color (coloring: np.ndarray, neighbors: set) -> int:
    """
    Helper function to assign the lowest available color not used by neighbors.
    """
    MAX_COLORS = 256 # Upper bound on number of colors
    used_colors = set()
    # Find used colors among neighbors
    for neighbor in neighbors:
        if coloring[neighbor] != -1:
            used_colors.add(coloring[neighbor])

    # Assign the lowest available color
    for color in range(MAX_COLORS):
        if color not in used_colors:
            return color

    assert False, "Exceeded maximum number of colors"
    

def graph_coloring (elements: np.ndarray) -> np.ndarray:
    """
    Graph coloring algorithm to assign parallelizable mesh elements into different color groups.

    Args:
        elements (np.ndarray [num_elements, num_vertices_per_element]): Array defining the mesh elements by their vertex indices.
    
    Returns: 
        np.ndarray [num_vertices]: Array of colors assigned to each vertex. -1 indicates uncolored.
    """
    MAX_COLORS = 256 # Upper bound on number of colors

    num_vertices = elements.max() + 1 # Assumes no missing vertices
    # Compute adjacency dict
    adjacency = {i: set() for i in range(num_vertices)}
    for ele in elements:
        for i in range(len(ele)):
            for j in range(i + 1, len(ele)):
                adjacency[ele[i]].add(ele[j])
                adjacency[ele[j]].add(ele[i])
    
    coloring = -1 * np.ones(num_vertices, dtype=int)
    for vertex in range(num_vertices):
        # Find used colors among neighbors
        used_colors = set()
        for neighbor in adjacency[vertex]:
            if coloring[neighbor] != -1:
                used_colors.add(coloring[neighbor])
        
        # Assign the lowest available color
        for color in range(MAX_COLORS):
            if color not in used_colors:
                coloring[vertex] = color
                break

    print(f"Assigned {coloring.max() + 1} colors.")
    print(f"Color distribution: {np.bincount(coloring)}")

    return coloring


def find_neighbors (adjacency: np.ndarray, vertex: int) -> list:
    """
    Helper function to find neighbors of a vertex from adjacency matrix.
    """
    neighbors = []
    for i in range(adjacency.shape[0]):
        if adjacency[vertex, i] > 0:
            neighbors.append(i)
    return neighbors

def graph_coloring_wp (elements: np.ndarray) -> np.ndarray:
    """
    Graph coloring algorithm to assign parallelizable mesh elements into different color groups.

    Args:
        elements (np.ndarray [num_elements, num_vertices_per_element]): Array defining the mesh elements by their vertex indices.
    
    Returns: 
        np.ndarray [num_vertices]: Array of colors assigned to each vertex. -1 indicates uncolored.
    """
    num_vertices = elements.max() + 1 # Assumes no missing vertices
    # Compute adjacency matrix
    adjacency = np.zeros([num_vertices, num_vertices], dtype=int)
    # parallel
    for ele in elements:
        for i, ele_i in enumerate(ele):
            # i and j are distinct
            for j in range(i + 1, len(ele)):
                # addition to prevent race conditions
                adjacency[ele_i, ele[j]] += 1
                adjacency[ele[j], ele_i] += 1

    coloring = -1 * np.ones(num_vertices, dtype=int)
    # parallel
    for vertex in range(num_vertices):
        coloring[vertex] = assign_color(coloring, adjacency[vertex])

    # Resolve conflicts
    MAX_ITER = 100
    iteration = 0
    vertices_inspect = np.arange(num_vertices)
    while len(vertices_inspect) > 0:
        # shared list of conflicts
        conflicts = []
        # parallel
        for vertex in vertices_inspect:
            neighbors = find_neighbors(adjacency, vertex)
            for neighbor in neighbors:
                if coloring[neighbor] == coloring[vertex]:
                    coloring[vertex] = assign_color(coloring, neighbors)
                    conflicts.append(vertex)
                    break
    
        vertices_inspect = list(set(conflicts))
        iteration += 1
        assert iteration < MAX_ITER, "Exceeded maximum number of conflict resolution rounds"

    print(f"Graph coloring resolved in {iteration-1} rounds.")
    return coloring
