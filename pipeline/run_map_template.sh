#!/bin/bash
# Copy into a new scene folder as run_map.sh. Maps every frame on both ComfyUI ports, then aligns them (LingBot-Map by default).
# Log: run_map.log, align.log.  Use ALIGNER=mapanything for the fallback.
cd "$(dirname "$0")"; mkdir -p scenes
echo "$(date +%T) mapping"
fs=(frames_960/*.png); half=$(( ${#fs[@]} / 2 ))
bash map_frames.sh 8188 "${fs[@]:0:$half}" > map_8188.log 2>&1 &
bash map_frames.sh 8190 "${fs[@]:$half}" > map_8190.log 2>&1 &
wait
for p in 8188 8190; do curl -s -X POST localhost:$p/free -H "Content-Type: application/json" -d '{"unload_models":true,"free_memory":true}'; done
echo "$(date +%T) mapped $(ls -d scenes/*/ | wc -l) frames, aligning"
sc=(scenes/*/); bash ../../pipeline/align_scenes.sh aligned "${sc[@]%/}" > align.log 2>&1
echo "$(date +%T) aligned exit $?, cleaning the map"
# Draft for the inventory step: one entry per real object plus the room outline. Under 8 frames it is flagged needs_review.
"$LOCALAPPDATA/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/Scripts/python.exe" ../../pipeline/clean_map.py aligned/alignment_good.json clean > clean.log 2>&1
echo "$(date +%T) ALL DONE exit $?"
