import math, random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def estimate_arboricity(G: nx.Graph) -> int:
    """Estimate arboricity α from the graph's degeneracy."""
    if G.number_of_edges() == 0:
        return 0
    # Degeneracy = max core number
    core_nums = nx.core_number(G)
    d = max(core_nums.values())
    # Arboricity is at least roughly half of degeneracy (degeneracy ≤ 2α - 1)
    alpha = (d + 2) // 2  # ceil((d+1)/2)
    return max(alpha, 1) if G.number_of_edges() > 0 else 0

def deterministic_coloring(G: nx.Graph) -> dict:
    """Deterministic coloring using a greedy strategy in degeneracy order."""
    coloring = {}
    if G.number_of_nodes() == 0:
        return coloring
    # Compute degeneracy ordering by iteratively removing min-degree vertices
    remaining = set(G.nodes())
    deg = {u: G.degree(u) for u in G.nodes()}
    removal_order = []
    while remaining:
        # Pick the vertex with smallest degree (tie-break by id for consistency)
        u = min(remaining, key=lambda x: (deg[x], x))
        remaining.remove(u)
        removal_order.append(u)
        for v in G.neighbors(u):
            if v in remaining:
                deg[v] -= 1
    # Color in reverse removal order (degeneracy order)
    for u in reversed(removal_order):
        # Gather neighbor colors (already colored neighbors only)
        used_colors = {coloring[v] for v in G.neighbors(u) if v in coloring}
        # Assign the smallest available color
        color = 0
        while color in used_colors:
            color += 1
        coloring[u] = color
    return coloring

def randomized_coloring(G: nx.Graph, max_iters: int = 10) -> dict:
    """Randomized coloring using graph shattering approach."""
    coloring = {}
    n = G.number_of_nodes()
    if n == 0:
        return coloring
    α = estimate_arboricity(G)
    # Palette size ≈ α * log2(α), ensure at least 2
    palette_size = max(2, int(α * math.log2(α)) if α > 1 else 2)
    # Initial random color assignment
    nodes = list(G.nodes())
    random_colors = np.random.randint(0, palette_size, size=len(nodes))
    coloring = {node: int(c) for node, c in zip(nodes, random_colors)}
    # Iteratively resolve conflicts by random reassignment
    for _ in range(max_iters):
        conflict_nodes = set()
        for u, v in G.edges():
            if coloring[u] == coloring[v]:
                conflict_nodes.add(u)
                conflict_nodes.add(v)
        if not conflict_nodes:
            break  # no conflicts
        for u in conflict_nodes:
            # Assign a new random color (different from current to ensure change)
            current = coloring[u]
            new_color = current
            if palette_size > 1:
                while new_color == current:
                    new_color = random.randrange(palette_size)
            coloring[u] = new_color
    # Final check and greedy fix for any remaining conflicts
    conflict_edges = [(u, v) for u, v in G.edges() if coloring.get(u) == coloring.get(v)]
    if conflict_edges:
        # Build conflict graph and find connected components of conflicts
        conflict_graph = nx.Graph()
        conflict_graph.add_edges_from(conflict_edges)
        for comp in nx.connected_components(conflict_graph):
            # Greedy-recoloring this component (list coloring with constraints)
            comp = list(comp)
            for u in comp:  # removing old colors for comp vertices
                del coloring[u]
            for u in comp:
                # Neighbor colors to avoid (both in comp if already colored, and outside comp)
                forbidden = {coloring[v] for v in G.neighbors(u) if v in coloring}
                color = 0
                while color in forbidden:
                    color += 1
                coloring[u] = color
    return coloring

# Example usage: generate a sparse random graph, apply both algorithms, and visualize.
if __name__ == "__main__":
    import matplotlib.colors as mcolors
    # Generate a random graph with up to 64 vertices (e.g., 30 vertices, 60 edges here)
    n, m = 20, 15 #30 60
    G = nx.gnm_random_graph(n, m, seed=42)  # fixed seed for reproducibility
    det_colors = deterministic_coloring(G)
    rand_colors = randomized_coloring(G)
    # Prepare color lists for plotting
    det_color_vals = [det_colors[node] for node in G.nodes()]
    rand_color_vals = [rand_colors[node] for node in G.nodes()]
    # Map color indices to actual color codes for visibility
    det_unique = sorted(set(det_color_vals)); rand_unique = sorted(set(rand_color_vals))
    det_cmap = {c: mcolors.to_hex(plt.cm.tab20(i % 20)) for i, c in enumerate(det_unique)}
    rand_cmap = {c: mcolors.to_hex(plt.cm.tab20(i % 20)) for i, c in enumerate(rand_unique)}
    det_node_colors = [det_cmap[c] for c in det_color_vals]
    rand_node_colors = [rand_cmap[c] for c in rand_color_vals]
    # # Draw the graph with NetworkX (side-by-side subplots for deterministic vs randomized)
    # pos = nx.spring_layout(G, seed=1)  # layout positions
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].set_title("Deterministic Coloring"); axes[1].set_title("Randomized Coloring")
    # nx.draw(G, pos, node_color=det_node_colors, edge_color='black', node_size=300, ax=axes[0])
    # nx.draw(G, pos, node_color=rand_node_colors, edge_color='black', node_size=300, ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    

    # Deterministic coloring in its own window
    plt.figure(figsize=(6,6))
    plt.title("Deterministic Coloring")
    nx.draw(
        G, pos,
        node_color=det_node_colors,
        edge_color='black',
        node_size=300
    )
    plt.show()

    # Randomized coloring in its own window
    plt.figure(figsize=(6,6))
    plt.title("Randomized Coloring")
    nx.draw(
        G, pos,
        node_color=rand_node_colors,
        edge_color='black',
        node_size=300
    )
    plt.show()