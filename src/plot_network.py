import os, warnings, andes
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.lines import Line2D

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


def plot_freq_and_rocof(ss, save_path="figures/freq_rocof.png", yidx=None):
    """
    Plot both frequency [Hz] and RoCoF [Hz/s] from an ANDES system and save the figure.
    """
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
    ss.TDS.load_plotter()

    # Generator names for legend
    df = ss.GENROU.as_df()
    # names = [f"Gen {g}, bus {b}" for g, b in zip(df["gen"], df["bus"])] # No need TODO: remove
    f_nom = ss.config.freq
    T = ss.TDS.plotter.get_values(0)
    yidx = ss.TDS.plotter.find("omega", idx_only=True) if not yidx else yidx

    fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Frequency
    ss.TDS.plotter.plot(
        yidx, ycalc=lambda y: f_nom * y,
        ylabel="Frequency [Hz]", # yheader=names,
        grid=True, legend=True, show=False, fig=fig, ax=axs[0]
    )
    axs[0].axhline(f_nom, ls="--", alpha=0.5, color="k")

    # RoCoF
    ss.TDS.plotter.plot(
        yidx, ycalc=lambda y: np.gradient(f_nom * y, T, axis=0),
        ylabel="RoCoF [Hz/s]", # yheader=names,
        grid=True, legend=True, show=False, fig=fig, ax=axs[1]
    )

    axs[0].set_title(f"{ss.name} — Frequency and RoCoF", fontsize=13, fontweight="bold")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    axs[0].set_ylim(49.5,50.1)
    axs[1].set_ylim(-1,1)
    plt.subplots_adjust(left=0.15)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved frequency/RoCoF plot to {save_path}")


# -----------------------------
# Quick test
# -----------------------------

def first_plots():
    andes.config_logger(stream_level=20)


    # names = [("ieee14/ieee14_solar.xlsx", "IEEE14 Solar"), ("ieee14/ieee14_full.xlsx", "IEEE14"),
    #          ("ieee39/ieee39_full.xlsx", "IEEE39"), ("kundur/kundur_full.xlsx", "Kundur")]


    ss = andes.load(andes.get_case("ieee14/ieee14_solar.xlsx"), name="IEEE14 Solar", setup=False)
    
    # val = ss.models
    # for key, line in val.items():
    #     if line.as_df().empty:
    #         continue
    #     print(key)
    #     print(line.as_df())
    #     print()

    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(5)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(8)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(2)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(3)))

    ss.setup()
    ss.PFlow.run(tol=1.00)
    ss.TDS.config.tf = float(10)
    ss.TDS.run()

    plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_L1_L17_L3.png")

    for layout in ["kamada"]:
        out_path = f"figures/network_{ss.name}_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)

    del(ss)

    ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), name="IEEE14", setup=False)
    
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1)))
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1.1)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(5)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(8)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(2)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(3)))

    ss.setup()

    # val = ss.models
    # for key, line in val.items():
    #     if line.as_df().empty:
    #         continue
    #     print(key)
    #     print(line.as_df())
    #     print()
    
    ss.PFlow.run(tol=1.00)
    ss.TDS.config.tf = float(10)
    ss.TDS.run()

    plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_L1_L17_L3.png")

    for layout in ["kamada"]:
        out_path = f"figures/network_{ss.name}_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)

    del(ss)

    ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), name="IEEE14 Modified with R5", setup=False)
    
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1)))
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1.1)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(5)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(8)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(2)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(3)))
    
    syg5 = 'GENROU_5'
    stg5 = ss.GENROU.get(src='gen', attr='v', idx=syg5)
    bus5 = ss.GENROU.get(src='bus', attr='v', idx=syg5)
    Sn5 = ss.GENROU.get(src='Sn', attr='v', idx=syg5)
    p05 = ss.StaticGen.get(src='p0', attr='v', idx=stg5)
    tg5 = ss.TGOV1.find_idx(keys='syn', values=[syg5])[0]
    R5 = ss.TGOV1.get(src='R', attr='v', idx=tg5)

    ss.add(model="REGCV1", param_dict=dict(
        bus=bus5,         # same bus as the PV
        gen=stg5,         # substitute the PV in TDS
        Sn=Sn5,
        M=ss.GENROU.get(src='M', attr='v', idx=syg5),         # virtual inertia [s]  (try 0, 4, 8, 12…)
        D=ss.GENROU.get(src='D', attr='v', idx=syg5),         # damping [pu/Hz]
        kw=1/R5,       # droop gain
        kv=0,
        ))

    ss.SynGen.alter(src='u', idx=syg5, value=0)

    ss.add("BusROCOF", dict(bus=8, Tf=0.02, Tw=0.02)) # Necessary to know the frequency at that bus (REGCV1)
    ss.add(model='BusROCOF', param_dict=dict(bus=1, Tr=0.5))


    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.pq2z = 0

    ss.setup()
    
    ss.PFlow.run()

    ss.TGOV1.alter(src='VMIN', idx=ss.TGOV1.idx.v, value=-10) # TODO: what is this for?

    ss.TDS.config.no_tqdm = True
    ss.TDS.config.criteria = 0
    ss.TDS.config.tf = float(10)
    _ = ss.TDS.init()
    ss.TDS.run()

    ss.TDS.load_plotter()
    plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_L1_L17_L3.png", yidx=[6,7,8,9,318])

    for layout in ["kamada"]:
        out_path = f"figures/network_{ss.name}_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)

    del(ss)

    ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), name="IEEE14 Modified", setup=False)
    
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1)))
    ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1.1)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(2)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(3)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(5)))
    ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(8)))
    
    syg5 = 'GENROU_5'
    stg5 = ss.GENROU.get(src='gen', attr='v', idx=syg5)
    bus5 = ss.GENROU.get(src='bus', attr='v', idx=syg5)
    Sn5 = ss.GENROU.get(src='Sn', attr='v', idx=syg5)
    p05 = ss.StaticGen.get(src='p0', attr='v', idx=stg5)
    tg5 = ss.TGOV1.find_idx(keys='syn', values=[syg5])[0]
    R5 = ss.TGOV1.get(src='R', attr='v', idx=tg5)

    ss.add(model="REGCV1", param_dict=dict(
        bus=bus5,         # same bus as the PV
        gen=stg5,         # substitute the PV in TDS
        Sn=Sn5,
        M=ss.GENROU.get(src='M', attr='v', idx=syg5),         # virtual inertia [s]  (try 0, 4, 8, 12…)
        D=ss.GENROU.get(src='D', attr='v', idx=syg5),         # damping [pu/Hz]
        kw=0,       # droop gain
        kv=0,
        ))

    ss.SynGen.alter(src='u', idx=syg5, value=0)

    ss.add("BusROCOF", dict(bus=8, Tf=0.02, Tw=0.02)) # Necessary to know the frequency at that bus (REGCV1)
    ss.add(model='BusROCOF', param_dict=dict(bus=1, Tr=0.5))

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.pq2z = 0

    ss.setup()
    
    ss.PFlow.run()

    ss.TGOV1.alter(src='VMIN', idx=ss.TGOV1.idx.v, value=-10) # TODO: what is this for?

    ss.TDS.config.no_tqdm = True
    ss.TDS.config.criteria = 0
    ss.TDS.config.tf = float(10)
    _ = ss.TDS.init()
    ss.TDS.run()

    print(ss.GENROU.as_df())
    ss.TDS.load_plotter()
    print(ss.TDS.plotter.find("omega", idx_only=True))
    plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_L1_L17_L3.png", yidx=[6,7,8,9,318])

    for layout in ["kamada"]:
        out_path = f"figures/network_{ss.name}_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)


if __name__ == "__main__":

    ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), name="IEEE14 Modified with R5", setup=False)
    
    ss.add("Toggle", dict(model="Line", dev="Line_4", t=float(3)))
    ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(5)))
    ss.add("Toggle", dict(model="SynGen", dev="GENROU_2", t=float(1)))
    
    syg5 = 'GENROU_5'
    stg5 = ss.GENROU.get(src='gen', attr='v', idx=syg5)
    bus5 = ss.GENROU.get(src='bus', attr='v', idx=syg5)
    Sn5 = ss.GENROU.get(src='Sn', attr='v', idx=syg5)
    p05 = ss.StaticGen.get(src='p0', attr='v', idx=stg5)
    tg5 = ss.TGOV1.find_idx(keys='syn', values=[syg5])[0]
    R5 = ss.TGOV1.get(src='R', attr='v', idx=tg5)

    ss.add(model="REGCV1", param_dict=dict(
        bus=bus5,         # same bus as the PV
        gen=stg5,         # substitute the PV in TDS
        Sn=Sn5,
        M=ss.GENROU.get(src='M', attr='v', idx=syg5),         # virtual inertia [s]  (try 0, 4, 8, 12…)
        D=ss.GENROU.get(src='D', attr='v', idx=syg5),         # damping [pu/Hz]
        kw=1/R5,       # droop gain
        kv=0,
        ))
    
    print(f"kw={1/R5}")

    ss.SynGen.alter(src='u', idx=syg5, value=0)
    ss.add("BusROCOF", dict(bus=8, Tf=0.02, Tw=0.02)) # Necessary to know the frequency at that bus (REGCV1)
    ss.add(model='BusROCOF', param_dict=dict(bus=7, Tr=0.5))
    ss.add(model='BusROCOF', param_dict=dict(bus=9, Tr=0.5))

    syg6 = 'GENROU_6'
    stg6 = ss.GENROU.get(src='gen', attr='v', idx=syg6)
    bus6 = ss.GENROU.get(src='bus', attr='v', idx=syg6)
    Sn6 = ss.GENROU.get(src='Sn', attr='v', idx=syg6)
    p06 = ss.StaticGen.get(src='p0', attr='v', idx=stg6)
    tg6 = ss.TGOV1.find_idx(keys='syn', values=[syg6])[0]
    R6 = ss.TGOV1.get(src='R', attr='v', idx=tg6)

    ss.add(model="REGCV1", param_dict=dict(
        bus=bus6,         # same bus as the PV
        gen=stg6,         # substitute the PV in TDS
        Sn=Sn6,
        M=ss.GENROU.get(src='M', attr='v', idx=syg6),         # virtual inertia [s]  (try 0, 4, 8, 12…)
        D=ss.GENROU.get(src='D', attr='v', idx=syg6),         # damping [pu/Hz]
        kw=1/R6,       # droop gain
        kv=0,
        ))
    
    print(f"kw2={1/R6}")

    ss.SynGen.alter(src='u', idx=syg6, value=0)
    ss.add("BusROCOF", dict(bus=4, Tf=0.02, Tw=0.02)) # Necessary to know the frequency at that bus (REGCV1)
    ss.add(model='BusROCOF', param_dict=dict(bus=5, Tr=0.5))
    
    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.pq2z = 0

    ss.setup()

    print(ss.BusFreq.as_df())
    print(ss.BusROCOF.as_df())
    
    ss.PFlow.run()

    ss.TGOV1.alter(src='VMIN', idx=ss.TGOV1.idx.v, value=-10) # TODO: what is this for?

    ss.TDS.config.no_tqdm = True
    ss.TDS.config.criteria = 0
    ss.TDS.config.tf = float(10)
    _ = ss.TDS.init()
    ss.TDS.run()

    ss.TDS.load_plotter()
    plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_gen_novi_norcv.png", yidx=[366,367])

    for layout in ["kamada"]:
        out_path = f"figures/network_{ss.name}_{layout}.png"
        plot_network(ss, layout=layout, save_path=out_path)

    # del(ss)

    # ss = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), name="IEEE14 Modified", setup=False)
    
    # ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1)))
    # ss.add("Toggle", dict(model="Line", dev="Line_1", t=float(1.1)))
    # ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(2)))
    # ss.add("Toggle", dict(model="Line", dev="Line_3", t=float(3)))
    # ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(5)))
    # ss.add("Toggle", dict(model="Line", dev="Line_17", t=float(8)))
    
    # syg5 = 'GENROU_5'
    # stg5 = ss.GENROU.get(src='gen', attr='v', idx=syg5)
    # bus5 = ss.GENROU.get(src='bus', attr='v', idx=syg5)
    # Sn5 = ss.GENROU.get(src='Sn', attr='v', idx=syg5)
    # p05 = ss.StaticGen.get(src='p0', attr='v', idx=stg5)
    # tg5 = ss.TGOV1.find_idx(keys='syn', values=[syg5])[0]
    # R5 = ss.TGOV1.get(src='R', attr='v', idx=tg5)

    # ss.add(model="REGCV1", param_dict=dict(
    #     bus=bus5,         # same bus as the PV
    #     gen=stg5,         # substitute the PV in TDS
    #     Sn=Sn5,
    #     M=ss.GENROU.get(src='M', attr='v', idx=syg5),         # virtual inertia [s]  (try 0, 4, 8, 12…)
    #     D=ss.GENROU.get(src='D', attr='v', idx=syg5),         # damping [pu/Hz]
    #     kw=0,       # droop gain
    #     kv=0,
    #     ))

    # ss.SynGen.alter(src='u', idx=syg5, value=0)

    # ss.add("BusROCOF", dict(bus=8, Tf=0.02, Tw=0.02)) # Necessary to know the frequency at that bus (REGCV1)
    # ss.add(model='BusROCOF', param_dict=dict(bus=1, Tr=0.5))

    # ss.PQ.config.p2p = 1
    # ss.PQ.config.q2q = 1
    # ss.PQ.config.p2z = 0
    # ss.PQ.config.q2z = 0
    # ss.PQ.pq2z = 0

    # ss.setup()
    
    # ss.PFlow.run()

    # ss.TGOV1.alter(src='VMIN', idx=ss.TGOV1.idx.v, value=-10) # TODO: what is this for?

    # ss.TDS.config.no_tqdm = True
    # ss.TDS.config.criteria = 0
    # ss.TDS.config.tf = float(10)
    # _ = ss.TDS.init()
    # ss.TDS.run()

    # print(ss.GENROU.as_df())
    # ss.TDS.load_plotter()
    # print(ss.TDS.plotter.find("omega", idx_only=True))
    # plot_freq_and_rocof(ss, save_path=f"figures/freq_rocof_{ss.name}_Toggle_L1_L17_L3.png", yidx=[6,7,8,9,318])

    # for layout in ["kamada"]:
    #     out_path = f"figures/network_{ss.name}_{layout}.png"
    #     plot_network(ss, layout=layout, save_path=out_path)