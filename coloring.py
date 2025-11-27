### Implement graph coloring for meshes
import numpy as np
import warp as wp

def graph_coloring (elements : np.ndarray) -> np.ndarray:
    """
    Graph coloring algorithm to assign parallelizable mesh elements into different color groups.

    Args:
        elements (np.ndarray [num_elements, num_vertices_per_element]): Array defining the mesh elements by their vertex indices.
    
    Returns: 
        np.ndarray [num_vertices]: Array of colors assigned to each vertex. -1 indicates uncolored.
    """
    MAX_COLORS = 256 # Upper bound on number of colors

    num_vertices = elements.max() + 1 # Assumes no missing vertices
    coloring = np.ones(num_vertices, dtype=int) * -1  # -1 means uncolored
    for ele in elements:
        # Gather existing colors
        element_colors = set()
        for vertex_idx in ele:
            if coloring[vertex_idx] != -1:
                element_colors.add(coloring[vertex_idx])
        # Color new vertices
        for vertex_idx in ele:
            if coloring[vertex_idx] == -1:
                c = 0
                for c in range(MAX_COLORS):
                    if c not in element_colors:
                        coloring[vertex_idx] = c
                        element_colors.add(c)
                        break
                
    assert coloring.min() != -1, "Graph coloring failed: some vertices remain uncolored."

    return coloring
