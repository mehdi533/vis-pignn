import itertools
import numpy as np
import re
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx

from .config import (DATA_DIR, H_SCALES, DROOP_SCALES, REN_SHARES, EVENTS)
from .andes_utils import load_system, apply_case_modifications

# ---------- metrics ----------
def infer_system_frequency(df):
    """
    Return (df_with_freq, freq_col_name, time_col_name).
    If 'f_sys' isn't present, compute f_sys_proxy from generator omega columns in p.u.
    """
    cols = list(df.columns)
    # time column: ANDES exports time as the first column in CSV (typically 't' or unnamed)
    t_col = cols[0]
    f_col = None
    # prefer a direct frequency column if it exists
    for c in cols:
        if re.search(r'\bf_(sys|bus|grid)\b', c, re.I):
            f_col = c
            break
    if f_col is not None:
        return df, f_col, t_col

    # else compute from machine speeds (omega in p.u.)
    omega_cols = [c for c in cols if re.search(r'\bomega\b', c, re.I)]
    if not omega_cols:
        raise RuntimeError(
            "No 'f_sys' and no 'omega' columns found in TDS export. "
            "Open the CSV and tell me the first 12 column names."
        )

    # detect nominal freq from context (fallback 50)
    f_nom_guess = 50.0
    for c in cols:
        if re.search(r'\bf_(\w+)\b', c, re.I):
            v = df[c].to_numpy()
            if np.isfinite(v).any():
                m = np.nanmean(v)
                if m > 55: f_nom_guess = 60.0
                break

    omega_mat = df[omega_cols].to_numpy()  # [T, M]
    f_sys_proxy = omega_mat.mean(axis=1) * f_nom_guess

    df2 = df.copy()
    df2['f_sys_proxy'] = f_sys_proxy
    return df2, 'f_sys_proxy', t_col

def compute_labels_from_timeseries(df):
    """
    Compute (nadir, min RoCoF, t_settle) from a TDS table.
    Uses f_sys if present, else a proxy built from omega.
    """
    df, f_col, t_col = infer_system_frequency(df)
    f = df[f_col].to_numpy()
    t = df[t_col].to_numpy()
    f_nom = 60.0 if np.nanmean(f) > 55 else 50.0

    f_nadir = float(np.nanmin(f))
    rocof = np.diff(f) / np.diff(t)
    rocof_min = float(np.nanmin(rocof))

    tol = 0.01
    idx = np.where(np.abs(f - f_nom) < tol)[0]
    t_settle = -1.0
    if idx.size:
        first = idx[0]
        if np.all(np.abs(f[first:] - f_nom) < tol):
            t_settle = float(t[first])
    return f_nadir, rocof_min, t_settle

# ---------- topology ----------
def extract_topology_graph(ss):
    """
    Build an undirected NetworkX graph from ss (buses + lines), robust to
    NumParam wrappers and schema variations (fb/tb, ib/jb, bus1/bus2).
    """
    import numpy as np
    import networkx as nx

    def arr(p):
        if p is None:
            return None
        return np.asarray(getattr(p, "v", p))

    def first_attr(obj, names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return arr(v)
        return None

    G = nx.Graph()

    # ---- Buses ----
    bus = getattr(ss, "Bus", None)
    if bus is None:
        raise RuntimeError("ANDES system has no Bus component.")
    u_bus = arr(bus.u)
    if u_bus is None:
        raise RuntimeError("Bus.u not found.")
    u_bus = u_bus.astype(bool)

    ids = first_attr(bus, ["idx"])  # optional
    Vn  = first_attr(bus, ["Vn", "Vb"])  # nominal voltage
    if ids is None:
        ids = np.arange(len(u_bus))
    for i in range(len(u_bus)):
        if not u_bus[i]:
            continue
        nid = int(ids[i])
        vnom = 1.0
        if Vn is not None and i < len(Vn):
            try:
                vnom = float(Vn[i])
            except Exception:
                vnom = 1.0
        G.add_node(nid, Vn=vnom)

    # ---- Lines/Branches ----
    # Try Line, then Branch
    edge_comp = getattr(ss, "Line", None) or getattr(ss, "Branch", None)
    if edge_comp is not None:
        u_line = arr(getattr(edge_comp, "u", None))
        if u_line is None:
            u_line = np.ones(len(arr(getattr(edge_comp, "x", [])) or []), dtype=bool)
        else:
            u_line = u_line.astype(bool)

        # Endpoint fields may be named differently across cases
        fb = first_attr(edge_comp, ["fb", "ib", "bus1"])
        tb = first_attr(edge_comp, ["tb", "jb", "bus2"])
        # Electrical strength proxies
        x  = first_attr(edge_comp, ["x"])                  # reactance
        b  = first_attr(edge_comp, ["b", "B"])             # susceptance

        # If we still don't have endpoints, just skip edges safely
        if fb is not None and tb is not None:
            nrow = min(len(u_line), len(fb), len(tb))
            for k in range(nrow):
                if not u_line[k]:
                    continue
                try:
                    u = int(fb[k]); v = int(tb[k])
                except Exception:
                    continue
                if u not in G:
                    G.add_node(u, Vn=1.0)
                if v not in G:
                    G.add_node(v, Vn=1.0)
                # weight heuristic: 1/|x| if available; else |b|; else 1
                w = 1.0
                if x is not None and k < len(x) and x[k] not in (None, 0):
                    try:
                        w = 1.0 / abs(float(x[k]))
                    except Exception:
                        w = 1.0
                elif b is not None and k < len(b):
                    try:
                        w = abs(float(b[k]))
                    except Exception:
                        w = 1.0
                G.add_edge(u, v, weight=w)

    # Fallback: ensure at least a connected structure
    if G.number_of_edges() == 0 and G.number_of_nodes() > 1:
        nodes = list(G.nodes())
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            G.add_edge(a, b, weight=1.0)

    print("edge fields:", 
      "fb" if hasattr(edge_comp, "fb") else "",
      "tb" if hasattr(edge_comp, "tb") else "",
      "ib" if hasattr(edge_comp, "ib") else "",
      "jb" if hasattr(edge_comp, "jb") else "",
      "bus1" if hasattr(edge_comp, "bus1") else "",
      "bus2" if hasattr(edge_comp, "bus2") else "")

    return G

# ---------- graph sample ----------
def build_graph_sample(ss, G, ts_df, meta):
    """
    Creates a torch_geometric.data.Data sample:
      x: [N, F] node features (Vn, inertia_local, load_local, gen_local)
      edge_index, edge_attr
      y: [3] (nadir, RoCoF, t_settle)
      meta: case-level features [H_scale, droop_scale, ren_share, event_size, is_gen_trip]
    """
    nodes = list(G.nodes())
    N = len(nodes)

    Vn = np.array([G.nodes[n].get('Vn', 1.0) for n in nodes], dtype=float).reshape(-1, 1)
    inertia_local = np.zeros((N, 1), dtype=float)
    load_local    = np.zeros((N, 1), dtype=float)
    gen_local     = np.zeros((N, 1), dtype=float)

    # Try to map generator inertia/Pm to buses
    GEN = getattr(ss, "GEN", None)
    if GEN is not None:
        bus_idx = getattr(GEN, "bus", None) or getattr(GEN, "ib", None)
        Hvals   = getattr(GEN, "H",   None)
        Pmvals  = getattr(GEN, "Pm",  None)
        def _arr(p): 
            if p is None: return None
            return p.v if hasattr(p, 'v') else p
        bus_idx, Hvals, Pmvals = map(_arr, (bus_idx, Hvals, Pmvals))
        if bus_idx is not None:
            for k, use in enumerate(GEN.u):
                if not use: continue
                b = int(bus_idx[k])
                if b in G:
                    j = nodes.index(b)
                    if Hvals is not None: inertia_local[j, 0] += float(Hvals[k])
                    if Pmvals is not None: gen_local[j, 0]    += float(Pmvals[k])

    # Loads
    for comp_name in ("ZIP", "PQ"):
        comp = getattr(ss, comp_name, None)
        if comp is not None and hasattr(comp, "P"):
            bus_idx = getattr(comp, "bus", None) or getattr(comp, "ib", None)
            Pvals   = getattr(comp, "P", None)
            if hasattr(bus_idx, 'v'): bus_idx = bus_idx.v
            if hasattr(Pvals,  'v'): Pvals  = Pvals.v
            if bus_idx is not None and Pvals is not None:
                for k, use in enumerate(comp.u):
                    if not use: continue
                    b = int(bus_idx[k])
                    if b in G:
                        j = nodes.index(b)
                        load_local[j, 0] += float(Pvals[k])

    X = np.hstack([Vn, inertia_local, load_local, gen_local]).astype(float)

    # Edges
    edges = []
    weights = []
    for u, v, d in G.edges(data=True):
        ui = nodes.index(u); vi = nodes.index(v)
        edges.append([ui, vi]); edges.append([vi, ui])
        w = d.get('weight', 1.0)
        weights.append(w); weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float).view(-1, 1) if weights else torch.empty((0,1), dtype=torch.float)

    # Targets
    f_nadir, rocof_min, t_settle = compute_labels_from_timeseries(ts_df)

    meta_vec = np.array([
        meta.get('H_scale', 1.0),
        meta.get('droop_scale', 1.0),
        meta.get('ren_share', 0.0),
        meta.get('event_size', 0.1),
        1.0 if meta.get('event_type','gen_trip') == 'gen_trip' else 0.0
    ], dtype=float)

    meta_tensor = torch.tensor(meta_vec, dtype=torch.float).view(1, -1)  # << key change

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([f_nadir, rocof_min, t_settle], dtype=torch.float).view(1, -1),
        meta=meta_tensor,  # << becomes [1, meta_dim]
    )

    return data

# ---------- Dataset ----------
class VISDataset(InMemoryDataset):
    def __init__(self, root, regenerate=False, transform=None, pre_transform=None):
        self._regen = regenerate
        super().__init__(root, transform, pre_transform)
        # after super().__init__, process() has run (if needed)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["vis_pignn_dataset.pt"]

    def process(self):
        # If we aren't regenerating and the file already exists, don't do work.
        if (not self._regen) and os.path.exists(self.processed_paths[0]):
            return

        # ---- Generate dataset (your existing code) ----
        samples = []
        case_id = 0
        for Hs, Ds, Rs, (etype, esize) in itertools.product(H_SCALES, DROOP_SCALES, REN_SHARES, EVENTS):
            case_id += 1
            ss = load_system()                 # reload fresh each case
            G  = extract_topology_graph(ss)
            ts = apply_case_modifications(ss, H_scale=Hs, droop_scale=Ds, ren_share=Rs,
                                          event_type=etype, event_size=esize)
            meta = dict(case=case_id, H_scale=Hs, droop_scale=Ds, ren_share=Rs,
                        event_type=etype, event_size=esize)
            data = build_graph_sample(ss, G, ts, meta)
            samples.append(data)

        data, slices = self.collate(samples)

        # ✅ Save exactly where PyG will later load from:
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
