#!/bin/bash
# Poll the volume until both experiments are complete or 2 hours pass.
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
M=../modalenv/.venv/bin/modal
PAIRS=$(.venv/bin/python -I -c 'import sys; sys.path.insert(0,"."); import brecphot as bp; print(" ".join(f"{c}_{i}" for c,l in bp.SUBSET.items() for i in l))')
for round in $(seq 1 40); do
  col=0; swp=0
  for p in $PAIRS; do for s in 100 101 102 103; do
    c=$(timeout 60 $M volume ls photonic-brec-rates column-pilot/$p/s$s 2>/dev/null | grep -c npz); col=$((col + c))
    w=$(timeout 60 $M volume ls photonic-brec-rates loss-sweep/$p/s$s 2>/dev/null | grep -c json); swp=$((swp + w))
  done; done
  echo "$(date -u +%H:%M) column npz $col / 18816 ; sweep json $swp / 1372"
  if [ "$col" -ge 18816 ] && [ "$swp" -ge 1372 ]; then echo DONE; exit 0; fi
  sleep 180
done
echo TIMEOUT
