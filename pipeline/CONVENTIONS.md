# Photo -> map pipeline (English port)

English port of the Spanish scripts in the original Spanish `scripts` folder (not published) (those stay untouched; do not edit,
move or delete them). Everything here is English: file names, identifiers, comments, console output,
JSON keys, output file names. Behavior and numbers must stay identical to the Spanish version, apart from
the documented changes below.

## Script mapping (old -> new)

| old (scripts\)                                   | new (pipeline\)        |
|--------------------------------------------------|------------------------|
| foto_libre.py                                    | photo_to_map.py        |
| sam_escena.py + medir_escena.py + vlm.py (HTTP)  | comfy_client.py        |
| objetos_libres.py                                | detect_objects.py      |
| galeria_libre.py                                 | gallery_match.py       |
| huellas.py (query part only)                     | fingerprints.py        |
| escala_referencias.py                            | fix_scale.py           |
| montar_simbolico.py                              | build_symbolic.py      |
| plano.py                                         | draw_plan.py           |
| generar_libre.py                                 | generate_models.py     |
| silueta.py                                       | silhouette.py          |
| pose_generados.py                                | fit_poses.py           |
| montar_libre.py                                  | build_scene.py         |

No `exec()` loading of other scripts: import plain modules (`from comfy_client import ...`). Scripts run
with the ComfyUI standalone python and `-I`, so each script must do
`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` before importing siblings.
Blender scripts (build_symbolic.py, build_scene.py) run inside Blender and import nothing from here.

## comfy_client.py API (shared by every script)

```python
DEFAULT_PORT = "8188"
def base_url(port) -> str                                   # http://127.0.0.1:<port>
def upload_image(path, port=DEFAULT_PORT) -> str            # name to use in LoadImage
def run_workflow(workflow: dict, port=DEFAULT_PORT, timeout=1800) -> dict   # waits, returns history entry
def output_dir(port) -> str                                 # same mapping the old code uses
def sam_masks(image_name, phrase, subfolder, port=DEFAULT_PORT, max_det=1) -> (list[np.ndarray bool], error_or_None)
def moge_points(image_name, subfolder, port=DEFAULT_PORT, fov_deg=0.0) -> np.ndarray  # same as correr_moge + leer_glb_posiciones
def ask_vlm(image_path, question, port=DEFAULT_PORT, max_len=400) -> str
```
Keep the exact workflows, node settings and parsing of the old code (e.g. SAM text "phrase:N" for N>1,
TextGenerate sampling_mode "off", SaveText format "txt").

## Output files in a scene folder

| old                         | new                          |
|-----------------------------|------------------------------|
| render.png (input copy)     | photo.png                    |
| mapa.json                   | map.json                     |
| mapa_sin_escala.json        | map_unscaled.json            |
| guias.png                   | guides.png                   |
| vista_superior.png          | (dropped; draw_plan makes plan.png) |
| plano.png                   | plan.png                     |
| mascaras_objetos.npz        | object_masks.npz             |
| moge_puntos.npz             | moge_points.npz              |
| recortes\objeto_N.png       | crops\object_N.png           |
| generados\objeto_N.glb      | generated\object_N.glb       |
| generados\objeto_N_pose.json / _calidad.json | generated\object_N_pose.json / _quality.json |
| _simbolico_3d.png           | _symbolic_render.png         |
| _simbolico_encima_sin_numeros.png | _symbolic_overlay_raw.png |
| simbolico_encima_foto.png   | symbolic_overlay.png         |
| escena_simbolica.blend      | symbolic_scene.blend         |
| render_3d.png / encima_foto.png / comparacion.png / escena.blend | render_3d.png / overlay.png / comparison.png / scene.blend |
| _pipeline.log               | _pipeline.log                |

## map.json keys

Top level: `W, H, fov_deg, fov_source, camera_height_m, cv_axes_in_floor, fx_px, objects, distances,
scale_applied`.
`fov_source` values: `"EXIF"`, `"given"`, `"guessed by MoGe"`.
`distances`: list of `{a, b, distance_m}`.
`scale_applied`: `{factor, source ("your measurement" | "references"), references: [{kind, id, ratio}]}`.

Object keys (old -> new):
nombre->name, tipo->type, frases->phrases, centro_m->center_m, base_z_m, alto_m->height_m,
ancho_m->width_m, fondo_m->depth_m, giro_grados->yaw_deg, frente_seguro->front_reliable,
distancia_camara_m->camera_distance_m, caja_imagen->image_box, apoyo_imagen->image_foot,
recorte->crop, pixeles->pixels, parecido_galeria->gallery_match {model, similarity},
modelo->model, alto_real_m->real_height_m, giro_por_apariencia->yaw_from_appearance,
calidad_silueta->silhouette_score, revisar->needs_review.

Added by orientation.py (Orient Anything V2, after detect_objects): `orientation {azimuth, polar, rotation,
n_dirs, confidence}`, `yaw_geometry_deg` (the old shape-based yaw), `facing_source` ("orient-anything" |
"geometry"), `dims_swapped` (width/depth swapped because the facing turned ~90 deg). `yaw_deg` is then the
Orient Anything facing. Boxes are placed with the measured visible face toward the camera and the body
extending away from it (not "behind its front").

## Documented changes vs the Spanish version
- The VLM is asked for the object name in English (short lowercase noun phrase, e.g. "truck",
  "pallet of boxes", "man in blue vest"). Person detection in fix_scale.py uses whole-word English
  matching: `\b(person|people|man|men|woman|women|worker|boy|girl|child|kid)\b`.
- fix_scale.py keeps: `--measure "1-2=2.5" | "3=1.8" | "3:width=1.2" (also :depth, :height) | "camera=1.6"`,
  comma separated, median; auto scale outside 0.6-1.6 is rejected; rescaling an already scaled map starts
  from map_unscaled.json.
- Gallery data under `GALLERY_DIR` (not published) is existing data with Spanish keys
  (`huellas.npz`, `<stem>.json` with `dims_finales`): read those names as they are, do not rename them.
- Text drawn on images is English.

## Room shell (added)
- `room_layout.py <scene> [port]` (standalone python): SAM "wall"/"ceiling"/"roof"/"floor" + MoGe points -> `map.json["room"]`
  `{walls [{id W#, start_m, end_m, length_m, height_m, room_side, assumed_parts_m}], ceiling_height_m, floor {visible_area_m2, outline_m}}`.
  Objects near a wall are lined up with it; a box through a wall first loses guessed depth, then moves
  (`yaw_before_room_deg`, `depth_before_room_m`, `center_before_room_m`, `fitted_to_walls`). Each object gets `wall_distances`.
  Writes `room_masks.png` for checking.
- `export_point_cloud.py <scene>` -> `point_cloud.ply` (+ `point_cloud.json` with the turn that aligns the main wall to the axes).
- `run_spatiallm.py <scene> [seed]` (venv in `SPATIALLM_DIR`, weights CC-BY-NC-4.0) -> `spatiallm.json`, `spatiallm.txt`.
- `compare_layouts.py <scene>` -> `layout_comparison.png/.json` (our walls vs SpatialLM's).
- `merge_layouts.py <scene>`: walls get `source` "photo" | "photo + SpatialLM" | "SpatialLM", `confirmed_by`;
  SpatialLM walls the photo sees through are dropped; adds `room.openings` (doors/windows), `room.size_m`,
  `room.merged_with`; the photo-only walls stay in `room.walls_photo_only`.

## Object tags and two-photo fusion
- `object_tags.py` (ComfyUI .venv python) adds `tags` to every object: `colors` (k-means in Lab, named, with share),
  `floor_sides_m`, `height_m`, `anchor_m` (median floor position of the object's FOOT: its points in the lowest
  30 cm, so a tree is anchored by its pot, a chair by its legs) and `fingerprint_row` into `object_tags.npz`
  (DINOv3 vectors, same model as fingerprints.py). Run it AFTER the scale is fixed (anchor_scale records it).
- `object_match.quality(a, b)`: look 0.35, colour 0.25, height 0.25, size 0.15 -> 0..1, position not used.
- `object_match.auto_landmarks`: only DISTINCTIVE pairs (quality >= 0.6 and >= 0.15 above the next candidate,
  both ways) may propose an alignment; look-alikes (a set of equal chairs) can only agree with it.
  An object completed to a known size (known_sizes.py) is anchored by its whole-body centre and its heading.
- `fuse_photos.py`: `"landmarks": "auto"` or `[[idA, idB, z_from, z_to], ...]`; objects are matched by
  quality x exp(-(feet apart / 0.8 m)^2) with one global assignment; `check_pairs` are reported, never used to align.
