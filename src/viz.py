import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.loader import DataLoader
from .config import DATA_DIR, BATCH_SIZE
from .model_pignn import PIGNN
from .dataset import VISDataset, extract_topology_graph
from .andes_utils import load_system, tds_to_dataframe

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "font.size": 11,
})

FIG_DIR = (DATA_DIR / "fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _save(fig, name):
    png = FIG_DIR / f"{name}.png"
    svg = FIG_DIR / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print("saved:", png.name, "|", svg.name)

# 1) Learning curves -----------------------------------------------------------
def plot_learning_curves(metrics_csv=DATA_DIR/"metrics.csv"):
    df = pd.read_csv(metrics_csv)
    # Train loss
    fig = plt.figure()
    plt.plot(df["epoch"], df["train_loss"].astype(float))
    plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.title("Training Loss")
    _save(fig, "train_loss")

    # Per-target val MSE
    for col, title in [("val_mse_nadir","Val MSE — Nadir"),
                       ("val_mse_rocof","Val MSE — RoCoF"),
                       ("val_mse_tsettle","Val MSE — Settling Time")]:
        fig = plt.figure()
        plt.plot(df["epoch"], df[col].astype(float))
        plt.xlabel("Epoch"); plt.ylabel(col); plt.title(title)
        _save(fig, f"{col}")

# 2) Pred vs True + error histograms ------------------------------------------
@torch.no_grad()
def eval_predictions(model_path=DATA_DIR/"pignn_best.pt", split="test"):
    ds = VISDataset(DATA_DIR, regenerate=False)
    n = len(ds)
    n_train = int(0.7*n); n_val = int(0.15*n)
    idx = slice(n_train+n_val, n) if split=="test" else (slice(0, n_train) if split=="train" else slice(n_train, n_train+n_val))
    subset = [ds[i] for i in range(*idx.indices(n))]

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim  = ds[0].x.size(1)
    meta_dim = ds[0].meta.size(-1) if hasattr(ds[0], 'meta') else 0
    model = PIGNN(in_dim=in_dim, meta_dim=meta_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []; y_pred = []
    for b in loader:
        b = b.to(device)
        p,_ = model(b)
        y_true.append(b.y.detach().cpu().numpy())
        y_pred.append(p.detach().cpu().numpy())
    y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)
    return y_true, y_pred

def plot_pred_vs_true(model_path=DATA_DIR/"pignn_best.pt", split="test"):
    y_true, y_pred = eval_predictions(model_path, split)
    names = ["Nadir [Hz]", "RoCoF [Hz/s]", "t_settle [s]"]
    for i, nm in enumerate(names):
        fig = plt.figure()
        plt.scatter(y_true[:,i], y_pred[:,i], s=18)
        mn = min(y_true[:,i].min(), y_pred[:,i].min())
        mx = max(y_true[:,i].max(), y_pred[:,i].max())
        plt.plot([mn, mx],[mn, mx])
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(f"{nm}: True vs Pred")
        _save(fig, f"pred_vs_true_{i}")

        # Error histogram
        err = y_pred[:,i] - y_true[:,i]
        fig = plt.figure()
        plt.hist(err, bins=20)
        plt.xlabel("Prediction error"); plt.ylabel("Count"); plt.title(f"{nm}: Error Histogram")
        _save(fig, f"error_hist_{i}")

# 3) Grid topology -------------------------------------------------------------
def plot_topology():
    ss = load_system()           # same case as training config
    G  = extract_topology_graph(ss)
    # Node scalars for color/size (use load or inertia as example)
    nodes = list(G.nodes())
    size = np.ones(len(nodes))
    color = np.array([G.nodes[n].get("Vn", 1.0) for n in nodes])
    pos = nx.spring_layout(G, seed=3)  # deterministic

    fig = plt.figure()
    nx.draw_networkx_nodes(G, pos, node_size=200*size, node_color=color, cmap=None)
    nx.draw_networkx_edges(G, pos, width=0.8)
    plt.axis("off"); plt.title("Grid Topology (node attribute: Vn)")
    _save(fig, "grid_topology")

# 4) Time-series overlays ------------------------------------------------------
def plot_timeseries_overlay():
    ss = load_system()
    ss.TDS.config.tf = 10.0
    ss.TDS.run()
    df = tds_to_dataframe(ss)

    # Try BusFreq columns (p.u.) or fallback to omega proxy
    fcols = [c for c in df.columns if c.startswith("f BusFreq ")]
    if len(fcols) >= 1:
        f_pu = df[fcols].to_numpy().mean(axis=1)
        f_nom = 60.0 if np.nanmean(f_pu) > 1.1 else 50.0
        f_hz = f_pu * f_nom
        t = df[df.columns[0]].to_numpy()
        fig = plt.figure()
        plt.plot(t, f_hz)
        plt.xlabel("Time [s]"); plt.ylabel("System frequency [Hz]"); plt.title("System Frequency (mean of BusFreq)")
        _save(fig, "timeseries_freq")
    else:
        # Attempt omega proxy
        cols = [c for c in df.columns if "omega" in c]
        if cols:
            omega = df[cols].to_numpy().mean(axis=1)
            f_nom = 60.0 if np.nanmean(omega) > 1.1 else 50.0
            f_hz = omega * f_nom
            t = df[df.columns[0]].to_numpy()
            fig = plt.figure()
            plt.plot(t, f_hz)
            plt.xlabel("Time [s]"); plt.ylabel("System frequency [Hz]"); plt.title("System Frequency (omega proxy)")
            _save(fig, "timeseries_freq")
        else:
            print("No frequency/omega columns found in TDS export.")

def main():
    plot_learning_curves()
    plot_pred_vs_true()
    plot_topology()
    plot_timeseries_overlay()
    print("All figures exported to:", FIG_DIR.as_posix())

if __name__ == "__main__":
    main()
