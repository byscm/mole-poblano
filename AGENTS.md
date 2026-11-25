# Notes for AI coding agents

This file is for coding agents (Claude Code, Cursor, Codex, Antigravity…) that a user opens on this repository to
adjust a room the pipeline already built. Read it before editing anything in a scene folder.

## What a scene folder is

`photo_to_map.py` writes one folder per room, usually `out/<room>/`. The source of truth is **`map.json`**; every
image and `.blend` in the folder is rebuilt from it. The full list of files and keys is in
`pipeline/CONVENTIONS.md`.

## The edit loop

1. Edit `map.json` (keep it valid JSON, keep the object `id`s unique, do not rename keys).
2. Rebuild with **one command** (Windows, ComfyUI Desktop python; ComfyUI does not need to be running):

   ```powershell
   & "$env:LOCALAPPDATA\Comfy-Desktop\ComfyUI-Installs\ComfyUI\standalone-env\python.exe" -I pipeline\rebuild.py out\<room>
   ```

   It recomputes the object-to-object and camera distances, rebuilds the Blender scene (with the 3D models when the
   folder has them, boxes otherwise; force one with `--symbolic` or `--3d`) and redraws `plan.png`. About 10–20 s.
3. Check `plan.png` (top view with sizes) and `render_3d.png` / `overlay.png` (or `symbolic_overlay.png`), and
   tell the user what changed with the new numbers.

Never re-run `photo_to_map.py` on a photo to "apply" an edit: that re-detects everything from scratch.

## Coordinates and units

- Metres. Floor plane, **camera at the origin** `(0, 0)`, standing `camera_height_m` above the floor.
- `+y` points away from the camera (into the photo), `+x` to the right as seen in the photo.
  So "20 cm to the left" is `center_m[0] -= 0.20`, "further back" is `center_m[1] += …`.
- `center_m [x, y]`: centre of the **side the camera sees**; the body extends `depth_m / 2` further away from the camera.
- `base_z_m`: height of the object's bottom above the floor (0 for things standing on the floor).
- `width_m`, `depth_m`, `height_m`: size of the object.
- `yaw_deg`: where the front faces, counter-clockwise seen from above. `0` = facing the camera, `90` = facing `+x`
  (right), `180` = facing away, `270` = facing left.
- Walls exist only when `map.json` has a `room` key (`room.walls[]` with `start_m`, `end_m`; see "Room shell" in
  `CONVENTIONS.md`). Without it, "away from the wall" cannot be measured: ask the user for a direction instead.

## What each edit really changes

| Edit | Boxes scene / plan | 3D scene with models |
|---|---|---|
| move (`center_m`, `base_z_m`) | yes | yes |
| `height_m` | yes | yes: a generated model is scaled uniformly to this height |
| `width_m`, `depth_m` | yes | **no**: generated models keep their own proportions; gallery models (`model` set) keep their real size |
| `yaw_deg` | yes, unless `yaw_from_appearance` exists (it wins: edit both) | **no** when `generated/object_<id>_pose.json` exists (the model is turned toward the camera from that pose) |
| delete an object | remove it from `objects` | same; its files in `generated/` can stay |

If the user wants a change that the 3D scene ignores, say so plainly instead of claiming it was done.

## Do not

- Do not run `photo_to_map.py <folder> --measure …` after hand edits: rescaling starts again from
  `map_unscaled.json` and **overwrites positions and sizes**. Fix the scale first, then edit.
- Do not edit `map_unscaled.json`, `object_masks.npz`, `moge_points.npz` or `crops/`; they belong to detection.
- Do not delete files from the user's folders; if something must go, move it to another folder and say where.
- This repository ships no API keys. Do not add any, and do not call paid services unless the user asks for it.
