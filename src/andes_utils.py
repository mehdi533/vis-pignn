import numpy as np
import pandas as pd
import andes
from andes.utils.paths import get_case

from .config import ANDES_CASE, T_FINAL, DT_HINT

def _as_array(p):
    """Return numpy array from ANDES NumParam or plain array; None -> None."""
    if p is None:
        return None
    return np.asarray(getattr(p, "v", p))

def _active_indices(comp):
    """Return indices of active rows for a component; [] if none."""
    u = _as_array(getattr(comp, "u", None))
    if u is None:
        return []
    return list(np.flatnonzero(u.astype(bool)))

def _maybe_scale_field(ss, comp_name, field, scale_fn):
    """
    For each active row in comp_name.field, set value <- scale_fn(old_value, idx).
    Returns True if anything was scaled; False otherwise.
    """
    comp = getattr(ss, comp_name, None)
    if comp is None or not hasattr(comp, field):
        return False
    vals = _as_array(getattr(comp, field, None))
    if vals is None or vals.size == 0:
        return False
    idxs = _active_indices(comp)
    if not idxs:
        return False
    for i in idxs:
        newv = float(scale_fn(vals[i], i))
        ss.set(comp_name, field, newv, idx=i)
    return True

def load_system():
    """
    Load an ANDES shipped example and solve PF.
    Returns a System object with TDS ready.
    """
    andes.config_logger(stream_level=20)  # INFO
    ss = andes.run(get_case(ANDES_CASE), default_config=True)
    # Configure TDS
    if DT_HINT is not None:
        ss.TDS.config.h = float(DT_HINT)
    ss.TDS.config.tf = float(T_FINAL)
    return ss

def _param_values(sys, comp_name, field):
    """
    Safely get a numpy array of parameter values from a component field.
    Works whether the field is a NumParam or plain ndarray.
    """
    comp = getattr(sys, comp_name, None)
    if comp is None or not hasattr(comp, field):
        return None, None
    p = getattr(comp, field)
    # NumParam has .v, .u (mask). Array may not.
    if hasattr(p, 'v'):
        vals = np.asarray(p.v, dtype=float)
        mask = np.asarray(comp.u, dtype=bool) if hasattr(comp, 'u') else np.ones_like(vals, dtype=bool)
    else:
        vals = np.asarray(p, dtype=float)
        mask = np.ones_like(vals, dtype=bool)
    return vals, mask

def _set_param_scaled(sys, comp_name, field, scale):
    """
    Multiply a numeric parameter by scale for all active rows (per-row sys.set to avoid .copy()).
    """
    comp = getattr(sys, comp_name, None)
    if comp is None or not hasattr(comp, field):
        return
    vals, mask = _param_values(sys, comp_name, field)
    if vals is None:
        return
    active_idx = np.flatnonzero(mask)
    for i in active_idx:
        newv = float(vals[i] * scale)
        sys.set(comp_name, field, newv, idx=i)

def _set_param_min(sys, comp_name, field, clip_to_field):
    """
    sys.comp.field = min(field, clip_to_field), per-row.
    """
    comp = getattr(sys, comp_name, None)
    if comp is None:
        return
    v1, m1 = _param_values(sys, comp_name, field)
    v2, m2 = _param_values(sys, comp_name, clip_to_field)
    if v1 is None or v2 is None:
        return
    active_idx = np.flatnonzero(m1 if m1 is not None else np.ones_like(v1, dtype=bool))
    for i in active_idx:
        sys.set(comp_name, field, float(min(v1[i], v2[i])), idx=i)

def _scale_param(sys, comp_name, field, scale):
    """
    sys.comp.field *= scale (per-row).
    """
    comp = getattr(sys, comp_name, None)
    if comp is None or not hasattr(comp, field):
        return
    vals, mask = _param_values(sys, comp_name, field)
    if vals is None:
        return
    for i in np.flatnonzero(mask):
        sys.set(comp_name, field, float(vals[i] * scale), idx=i)

def tds_to_dataframe(ss, csv_path=None):
    """
    Return a pandas DataFrame of the last TDS run.
    Works across ANDES versions: tries in-memory plotter df, else exports CSV.
    """
    ss.TDS.load_plotter()
    if hasattr(ss.TDS.plt, "df") and ss.TDS.plt.df is not None:
        return ss.TDS.plt.df.copy()
    out = csv_path or "_tmp_tds.csv"
    ss.TDS.plt.export_csv(out)
    return pd.read_csv(out)

def apply_case_modifications(ss, H_scale=1.0, droop_scale=1.0, ren_share=0.0,
                             event_type='gen_trip', event_size=0.1):
    # 1) Inertia H
    _maybe_scale_field(ss, "GEN", "H", lambda old, _: old * float(H_scale))

    # 2) Droop scaling (optional)
    droop_scaled = False
    candidates = [
        ("TGOV1", "R"), ("IEESGO", "R"), ("GOV", "R"),
        ("GOVSTEAM", "R"), ("REG", "R"),
        ("TGOV1", "droop"), ("GOV", "droop"),
    ]
    for comp_name, field in candidates:
        ok = _maybe_scale_field(ss, comp_name, field, lambda old, _: old / float(droop_scale))
        if ok:
            droop_scaled = True
            break

    # 3) Renewable share: scale Pmax, clip Pm
    _maybe_scale_field(ss, "GEN", "Pmax", lambda old, _: old * (1.0 - float(ren_share)))
    # clip Pm <= Pmax
    comp = getattr(ss, "GEN", None)
    if comp is not None and hasattr(comp, "Pm") and hasattr(comp, "Pmax"):
        Pm   = _as_array(comp.Pm)
        Pmax = _as_array(comp.Pmax)
        for i in _active_indices(comp):
            ss.set("GEN", "Pm", float(min(Pm[i], Pmax[i])), idx=i)

    # 4) Disturbance
    if event_type == "gen_trip":
        _maybe_scale_field(ss, "GEN", "Pm", lambda old, _: old * (1.0 - float(event_size)))
    elif event_type == "load_step":
        for load_comp in ("ZIP", "PQ"):
            _maybe_scale_field(ss, load_comp, "P", lambda old, _: old * (1.0 + float(event_size)))

    # 5) Run TDS
    ss.TDS.config.tf = float(T_FINAL)
    ss.TDS.run()
    return tds_to_dataframe(ss)
