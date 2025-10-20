#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.2
# choose the index that matches your torch version:
CPU_HTML="https://data.pyg.org/whl/torch-2.2.0+cpu.html"
python -m pip install torch-scatter -f ${CPU_HTML}
python -m pip install torch-sparse  -f ${CPU_HTML}
python -m pip install torch-cluster -f ${CPU_HTML}
python -m pip install torch-spline-conv -f ${CPU_HTML}
python -m pip install torch-geometric==2.7.0
python - <<'PY'
import torch, torch_geometric
print("OK: torch", torch.__version__, "pyg", torch_geometric.__version__)
PY
