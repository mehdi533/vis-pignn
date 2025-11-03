import os
import andes
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

# -----------------------------
# Build graph
# -----------------------------
def build_graph(ss: andes.System) -> nx.Graph:
    G = nx.Graph()

    # --- Buses ---
    bus_model = ss.models.get("Bus", None)
    bus_df = bus_model.as_df()

    assert bus_model
    assert not bus_df.empty 

    for _, row in bus_df.iterrows():
        G.add_node(int(row["idx"]), name=row["name"])

    # --- Lines ---
    line_model = ss.models.get("Line", None)
    line_df = line_model.as_df()

    assert line_model
    assert not line_df.empty

    for _, row in line_df.iterrows():
        b1, b2 = row["bus1"], row["bus2"]
        if b1 in G.nodes and b2 in G.nodes:
            G.add_edge(b1, b2)

    return G


# -----------------------------
# Layouts
# -----------------------------
def calculate_positions(G: nx.Graph, layout: str = "spring"):
    """
    Compute node positions based on the chosen layout.
    Available: spring, circular, kamada, shell
    """
    if G.number_of_nodes() == 0:
        return {}

    layout = layout.lower()

    if layout == "spring":
        return nx.spring_layout(G, seed=42, k=0.8)
    elif layout == "circular":
        return nx.circular_layout(G)
    elif layout == "kamada":
        return nx.kamada_kawai_layout(G)
    elif layout == "shell":
        return nx.shell_layout(G)
    else:
        print(f"⚠️ Unknown layout '{layout}', using kamada.")
        return nx.kamada_kawai_layout(G)


# -----------------------------
# Plotting
# -----------------------------
def draw_buses(ax, pos, ss):
    """
    Compact version of bus drawing logic.
    Handles PQ, PV, Slack, Renewables, and Shunts.
    Supports PQ+PV hybrid coloring.
    Returns dynamic legend handles.
    """

    # --- Extract relevant sets ---
    def get_buses(name):
        model = ss.models.get(name)
        return set(model.as_df()["bus"]) if model and not model.as_df().empty else set()

    pq_buses = get_buses("PQ")
    pv_buses = get_buses("PV")
    slack_buses = get_buses("Slack")
    shunt_buses = get_buses("Shunt")

    renewable_buses = set()
    for name in ["REGCV1", "REGCV2", "REGF1", "REGF2", "REGF3", "REGCA1", "REGCP1"]:
        renewable_buses |= get_buses(name)

    # --- Draw each bus ---
    for n, (x, y) in pos.items():
        # base
        color, shape, text_color = "lightblue", "circle", "black"

        if n in slack_buses:
            color, text_color = "black", "white"
        elif n in pq_buses and n in pv_buses:
            # Half red/half green (diagonal split)
            tri1 = Path([[x - 0.04, y - 0.04], [x + 0.04, y - 0.04], [x + 0.04, y + 0.04]])
            tri2 = Path([[x - 0.04, y - 0.04], [x - 0.04, y + 0.04], [x + 0.04, y + 0.04]])
            ax.add_patch(mpatches.PathPatch(tri1, fc="red", ec="black", lw=1, zorder=3))
            fc = "lightgreen" if n in renewable_buses else "green"
            ax.add_patch(mpatches.PathPatch(tri2, fc=fc, ec="black", lw=1, zorder=3))
            ax.add_patch(mpatches.Rectangle((x-0.04, y-0.04), width=0.08, height=0.08, color="None", ec="black", lw=1, zorder=3))
            text_color = "black"
        elif n in pq_buses:
            color, shape, text_color = "red", "square", "black"
        elif n in pv_buses:
            color, shape, text_color = "green", "square", "black"
        elif n in renewable_buses:
            color, shape = "#90ee90", "square"

        if n not in (pq_buses & pv_buses):  # we already drew the hybrid manually
            if shape == "circle":
                patch = mpatches.Circle((x, y), 0.05, color=color, ec="black", lw=1, zorder=3)
            else:
                color = "lightgreen" if n in renewable_buses else color
                patch = mpatches.Rectangle((x-0.04, y-0.04), width=0.08, height=0.08, color=color, ec="black", lw=1, zorder=3)
            ax.add_patch(patch)

        ax.text(x, y, str(n), ha="center", va="center", fontsize=7, fontweight="bold", color=text_color, zorder=4)

        # Shunt marker
        if n in shunt_buses:
            ax.add_patch(mpatches.Circle((x+0.08, y), 0.03, fc="gray", ec="black"))
            ax.text(x+0.08, y, "S", ha="center", va="center", fontsize=5, color="white", fontweight="bold")
            
    # --- Dynamic legend ---
    from matplotlib.lines import Line2D

    legends = [mpatches.Patch(color='lightblue', ec='black', label='Bus')]
    if pq_buses: legends.append(mpatches.Patch(color='red', ec='black', label='PQ (Load)'))
    if pv_buses: legends.append(mpatches.Patch(color='green', ec='black', label='PV (Gen)'))
    if renewable_buses: legends.append(mpatches.Patch(color="lightgreen", ec='black', label='Renewable Gen'))
    if slack_buses: legends.append(Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='black', markeredgecolor='black',
                                   markersize=8, label='Slack Bus'))
    if shunt_buses: legends.append(Line2D([0],[0], marker='o', linestyle='None', markerfacecolor='gray', markeredgecolor='black',
                                   markersize=8, label='Shunt (S)'))

    return legends


def plot_network(ss: andes.System, layout="spring", save_path="figures/network.png"):
    G = build_graph(ss)
    print(f"Graph: {G.number_of_nodes()} buses, {G.number_of_edges()} lines")

    if G.number_of_nodes() == 0:
        print("✗ Nothing to plot.")
        return

    pos = calculate_positions(G, layout)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw lines
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.2, edge_color="gray", alpha=0.7)
    
    # Draw buses
    line_patch = mpatches.Patch(color='gray', label='Transmission line')
    legend_handles = [line_patch]
    legend_handles.extend(draw_buses(ax, pos, ss))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{ss.name} power system network", fontsize=14)

    # --- Legend ---
    ax.legend(
        handles=legend_handles,
        loc='best',
        # bbox_to_anchor=(1.2, 0.95),
        fontsize=7,
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1.2, 1])  # leave space on right for legend
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved network plot ({layout}) to {save_path}")
    plt.close(fig)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    andes.config_logger(stream_level=20)
    ss = andes.load(andes.get_case("ieee14/ieee14_solar.xlsx"), name="IEEE14 Solar")
    
    # val = ss.models
    # for key, line in val.items():
    #     if line.as_df().empty:
    #         continue
    #     print(key)
    #     print(line.as_df())
    #     print()

    # Test all layouts
    for layout in ["spring", "circular", "kamada", "shell"]:
        out_path = f"figures/network_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)
