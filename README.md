# VIS-PIGNN: Physics-Informed GNN surrogate for frequency metrics (ANDES + PyTorch Geometric)

This repo generates dynamic grid data with **ANDES**, builds graph samples, and trains a **Physics-Informed Graph Neural Network (PIGNN)** to predict system-level frequency metrics:
- **Nadir** (Hz)
- **RoCoF** (Hz/s)
- **Settling time** (s)

It also exports clean, presentation-ready plots (learning curves, pred-vs-true, error histograms, topology, and time-series overlays).

---

## Repo layout

```
vis-pignn/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ scripts/
│  ├─ install_pyg_cpu.sh        # helper for torch-geometric CPU wheels
│  └─ quickcheck_andes.py       # sanity check: ANDES → TDS → CSV columns
├─ data/                        # generated (ignored) – datasets, figs, metrics, checkpoints
└─ src/
   ├─ config.py                 # case selection, sweeps, training knobs, paths
   ├─ andes_utils.py            # load case, parameter edits, TDS→DataFrame helper
   ├─ dataset.py                # topology extraction, labeling, PyG dataset
   ├─ model_pignn.py            # SAGE/GCN + physics regularizer
   ├─ train.py                  # dataset build, train/val/test, metrics.csv, best.pt
   └─ viz.py                    # plotting (loss curves, pred-vs-true, topology, timeseries)
```

---

## Quick start

### 0) Environment

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
bash scripts/install_pyg_cpu.sh   # installs torch-scatter/sparse/cluster/spline-conv for CPU
```

> If you have CUDA, follow the official PyG install matrix and adjust the wheel index in `scripts/install_pyg_cpu.sh`.

### 1) Sanity check ANDES

```bash
python scripts/quickcheck_andes.py
```

You should see PF + TDS finishing and a CSV exported. If there’s **no dynamics**, switch the case in `src/config.py` (e.g., `kundur/kundur_full.xlsx` or `ieee39/ieee39_full.xlsx`).

### 2) Train

```bash
python -m src.train
```

Outputs:
- processed dataset at `data/processed/vis_pignn_dataset.pt`
- best model at `data/pignn_best.pt`
- training metrics at `data/metrics.csv`

### 3) Visualize

```bash
python -m src.viz
```

Figures go to `data/fig/` as **PNG** and **SVG**:
- `train_loss.*`, `val_mse_nadir.*`, `val_mse_rocof.*`, `val_mse_tsettle.*`
- `pred_vs_true_*.*`
- `error_hist_*.*`
- `grid_topology.*`
- `timeseries_freq.*`

SVGs are slide/LaTeX friendly.

---

## Configuration

Open `src/config.py`:

- **ANDES_CASE**: stock cases in `andes/cases/` (e.g., `ieee39/ieee39_full.xlsx`, `kundur/kundur_full.xlsx`)
- **T_FINAL**: TDS simulation horizon (s)
- Parameter sweeps: `H_SCALES`, `DROOP_SCALES`, `REN_SHARES`, `EVENTS`
- Model/training knobs: `HIDDEN`, `LR`, `WD`, `EPOCHS`, `BATCH_SIZE`, `LAMBDA_LAP`

---

## How it works

1. **Data generation** (`andes_utils.py`)
   - Loads the case, scales **H**, **droop** (if governor fields exist), **Pmax/Pm** for renewables, and applies a simple disturbance (`gen_trip` or `load_step`).
   - Runs **TDS** and materializes a DataFrame via the plotter (`ss.TDS.load_plotter()` → in-memory or CSV fallback).

2. **Dataset build** (`dataset.py`)
   - Extracts **topology** (works with `bus1/bus2` or `fb/tb`/`ib/jb`) and computes edge weights (`1/|x|`, else `|b|`, else fallback).
   - Node features: `[Vn, ΣH(bus), ΣLoadP(bus), ΣPm(bus)]`.
   - Targets computed from time-series: **nadir**, **RoCoF**, **t_settle**. Uses `f BusFreq *` if present or builds a proxy from generator **omega**.
   - Case-level **meta** per graph: `[H_scale, droop_scale, ren_share, event_size, is_gen_trip]`.

3. **Model** (`model_pignn.py`)
   - 2-layer **SAGE** (or **GCN**) → global mean pool → MLP head.
   - **Physics regularizer**: Laplacian smoothness on node embeddings (sum over edges `w_ij ||h_i - h_j||^2`).

4. **Training** (`train.py`)
   - Random split (70/15/15), Adam, logs **metrics.csv**, saves `pignn_best.pt` on best val score.

5. **Visualization** (`viz.py`)
   - Learning curves, scatter: **True vs Pred**, error histograms, **grid topology**, system-frequency overlay.

---

## Troubleshooting

- **`torch-geometric` install errors**: use the wheel index in `scripts/install_pyg_cpu.sh`, or the official CUDA matrix for your Torch/CUDA combo.
- **`No differential equation detected.`**: the case has no dynamics. Switch to a dynamic case (`*_full.xlsx`) in `config.py`.
- **`ss.TDS.df` not found**: handled via the plotter (`tds_to_dataframe`). No action needed.
- **Governor `R` not found / index error**: droop scaling is optional. The code skips scaling if the field is absent.
- **Shape mismatches for `meta` or `y`**: dataset stores them as **[1, d]** per graph; batching yields **[B, d]**. If you changed this, ensure the model reads the correct `meta_dim` (`ds[0].meta.size(-1)`).

---

## Reproducibility

- Seed is set in `src/config.py` and used for splits.
- Each case reloads a **fresh ANDES system** to avoid cumulative parameter edits.

---

## License & citation

- Check ANDES’s license for redistribution constraints.
- If you use this code in a publication, please cite ANDES and PyG appropriately.

---

## Roadmap (nice-to-haves)

- z-score normalization for features/targets (improves nadir RMSE further)
- richer edge features (|z|, angle) & residual GNN
- scheduled toggles per generator/line (improves spatial learning)
- permutation feature importance for interpretability
