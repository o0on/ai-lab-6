import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2
    },
    "Patient_Wing": {
        "Laboratory": 5.0
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },
    "Laboratory": {
        "Emergency_Ward": 3.2
    },
    "Emergency_Ward": {}
}

def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def reconstruct_path(came_from, current):
    path = []
    curr = current
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path

def calculate_path_cost(path, graph):
    total_cost = 0.0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        total_cost += graph[u][v]
    return total_cost

def greedy_best_first_search(start, goal, graph):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    visited = set()
    came_from = {start: None}
    expansion_order = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = calculate_path_cost(path, graph)
            return path, expansion_order, cost

        for neighbour in graph.get(current, {}):
            if neighbour not in visited:
                if neighbour not in came_from:
                    came_from[neighbour] = current
                heapq.heappush(frontier, (heuristic(neighbour, goal), neighbour))

    return None, expansion_order, float('inf')

def a_star_search(start, goal, graph):
    frontier = []
    start_h = heuristic(start, goal)
    heapq.heappush(frontier, (start_h, 0.0, start))
    came_from = {start: None}
    g_costs = {start: 0.0}
    visited = set()
    expansion_order = []

    while frontier:
        f_cost, g_cost, current = heapq.heappop(frontier)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = g_cost
            return path, expansion_order, cost

        for neighbour, cost_val in graph.get(current, {}).items():
            tentative_g_cost = g_costs[current] + cost_val

            if neighbour not in g_costs or tentative_g_cost < g_costs[neighbour]:
                came_from[neighbour] = current
                g_costs[neighbour] = tentative_g_cost
                f_neighbor = tentative_g_cost + heuristic(neighbour, goal)
                heapq.heappush(frontier, (f_neighbor, tentative_g_cost, neighbour))

    return None, expansion_order, float('inf')

st.set_page_config(page_title="Hospital Robot Search Visualizer", layout="wide")

st.title("Autonomous Emergency Supply Robot: Pathfinding")
st.write("Compare Greedy Best-First Search (GBFS) and A* search across the hospital corridor network.")

nodes = list(hospital_graph.keys())

col1, col2, col3 = st.columns(3)

with col1:
    start = st.selectbox(
        "Select Initial Node",
        nodes,
        index=nodes.index("Pharmacy")
    )

with col2:
    goal = st.selectbox(
        "Select Goal Node",
        nodes,
        index=nodes.index("Emergency_Ward")
    )

with col3:
    algorithm = st.selectbox(
        "Select Search Algorithm",
        ["GBFS", "A*"]
    )

if st.button("Run Search"):
    if start == goal:
        st.warning("Initial node and Goal node must be distinct.")
    else:
        if algorithm == "GBFS":
            path, expansion_order, cost = greedy_best_first_search(start, goal, hospital_graph)
        else:
            path, expansion_order, cost = a_star_search(start, goal, hospital_graph)

        if path is None:
            st.error(f"No path found from {start} to {goal} using {algorithm}.")
        else:
            st.subheader("Search Result")
            st.write(f"**Algorithm:** {algorithm}")
            st.write(f"**Solution Path:** {' → '.join(path)}")
            st.write(f"**Total Path Cost:** {cost:.2f}")

            G = nx.DiGraph()
            for node, neighbors in hospital_graph.items():
                for neighbor, weight in neighbors.items():
                    G.add_edge(node, neighbor, weight=weight)

            pos = locations

            fig, ax = plt.subplots(figsize=(10, 6))

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#D9EAF7", edgecolors="#1B4965", node_size=2500, linewidths=2)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold", font_family="sans-serif")
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#A0AAB2", width=1.5, arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.05")

            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

            nx.draw_networkx_nodes(G, pos, nodelist=path, ax=ax, node_color="#FFD166", edgecolors="#E76F51", node_size=2500, linewidths=2.5)
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax, edge_color="#E63946", width=3.5, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.05")

            ax.set_title(f"{algorithm} Solution Path")
            ax.axis("off")

            st.pyplot(fig)