#!/bin/bash
# Default aligner for a scene's mapped frames. LingBot-Map unless ALIGNER=mapanything (Apache weights, the fallback).
# Usage: [ALIGNER=lingbot|mapanything] bash align_scenes.sh <out_folder> <scene_folder> ...
# Writes <out_folder>/alignment.json, alignment_good.json and vggt_points.npz, the files merge_photos.py and the plan tools read.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$1"; shift
mapfile -t SCENES < <(printf '%s\n' "${@%/}" | sort)
PY="$LOCALAPPDATA/Comfy-Desktop/ComfyUI-Installs/ComfyUI/standalone-env/python.exe"
mkdir -p "$OUT"
if [ "${ALIGNER:-lingbot}" = "mapanything" ]; then
  "${MAPANYTHING_PYTHON:?set MAPANYTHING_PYTHON}" "$HERE/mapanything_align.py" "$OUT" "${SCENES[@]}"
  "$PY" -I -c "
import json, sys
d = json.load(open(sys.argv[1] + '/alignment.json'))
d['photos'] = {k: v for k, v in d['photos'].items() if v['tilt_error_deg'] <= 20 and abs(v['camera_height_from_vggt_m'] - v['camera_height_own_m']) <= 0.6}
json.dump(d, open(sys.argv[1] + '/alignment_good.json', 'w'), indent=1); print('KEPT', len(d['photos']))" "$OUT"
  exit 0
fi
mkdir -p "$OUT/input"
for s in "${SCENES[@]}"; do cp "$s/photo.png" "$OUT/input/$(basename "$s").png"; done
"${LINGBOT_PYTHON:?set LINGBOT_PYTHON}" "$HERE/lingbot_run.py" \
  --weights "${LINGBOT_WEIGHTS:?set LINGBOT_WEIGHTS to lingbot-map.safetensors}" --out "$OUT" --images "$OUT/input" --mode streaming --sdpa
"$PY" -I "$HERE/lingbot_align.py" "$OUT/pred.npz" "$OUT" "${SCENES[@]}"
