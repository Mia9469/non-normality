#!/usr/bin/env bash
# Create the .venv for the Shiu/Brian2 model.
#
# RUN THIS ON THE MACHINE WHERE YOU WILL SIMULATE (the server) -- virtualenvs
# are NOT portable across OS / Python builds, so do not copy a Mac .venv to Linux.
#
# Needs Python 3.10 and a C/C++ compiler (gcc/clang) for Brian2 codegen /
# cpp_standalone. On the server:  module load gcc   (or ensure gcc is on PATH).
set -euo pipefail
PY=${PYTHON:-python3.10}
command -v "$PY" >/dev/null || { echo "ERROR: $PY not found. Set PYTHON=/path/to/python3.10"; exit 1; }

"$PY" -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
bash setup_critical_init.sh
python -c "import brian2, numpy, pandas, pyarrow, scipy, torch, rastermap; print('brian2', brian2.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__, '-> OK')"
echo
echo "Environment ready. Activate with:"
echo "    source .venv/bin/activate"
