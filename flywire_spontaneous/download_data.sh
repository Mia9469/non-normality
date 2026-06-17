#!/usr/bin/env bash
# Atomic, validated download of the Shiu et al. FlyWire model inputs.
set -euo pipefail

cd "$(dirname "$0")"
BASE="https://edmond.mpdl.mpg.de/api/access/datafile"
CON="2023_03_23_connectivity_630_final.parquet"
COMP="2023_03_23_completeness_630_final.csv"

validate_parquet() {
  python3 - "$1" <<'PY'
import sys
import pyarrow.parquet as pq
path = sys.argv[1]
meta = pq.read_metadata(path)
if meta.num_rows < 1:
    raise SystemExit(f"{path}: empty parquet")
print(f"{path}: valid parquet, rows={meta.num_rows}, row_groups={meta.num_row_groups}")
PY
}

download_atomic() {
  local id="$1"
  local target="$2"
  local tmp="${target}.part"
  rm -f "$tmp"
  curl -L --fail --retry 5 --retry-all-errors "$BASE/$id" -o "$tmp"
  mv "$tmp" "$target"
}

if ! validate_parquet "$CON" 2>/dev/null; then
  echo "downloading complete connectivity file..."
  download_atomic 214091 "$CON"
  validate_parquet "$CON"
else
  validate_parquet "$CON"
fi

if [[ ! -s "$COMP" ]]; then
  echo "downloading completeness file..."
  download_atomic 214095 "$COMP"
fi

echo "ready:"
ls -lh "$CON" "$COMP"
