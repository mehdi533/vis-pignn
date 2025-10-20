from pathlib import Path

SEED = 7

# ANDES case to load (use shipped examples)
# Options: 'ieee39/ieee39.xlsx' or 'kundur/kundur_full.xlsx'
ANDES_CASE = 'ieee39/ieee39_full.xlsx'  # instead of ieee39/ieee39.xlsx

# Simulation
T_FINAL = 10.0     # seconds
DT_HINT = None     # keep None to let ANDES choose; or set e.g. 0.01

# Parameter sweeps (keep small first; expand later)
H_SCALES      = [0.7, 1.0, 1.3]
DROOP_SCALES  = [0.8, 1.0, 1.2]
REN_SHARES    = [0.0, 0.3, 0.6]
EVENTS        = [('gen_trip', 0.05), ('gen_trip', 0.10), ('load_step', 0.08)]

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_PT = DATA_DIR / "vis_pignn_dataset.pt"

# Model / training
HIDDEN = 64
LR = 1e-3
WD = 1e-5
EPOCHS = 50
BATCH_SIZE = 8
LAMBDA_LAP = 1e-3
