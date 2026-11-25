# Second, independent room plan: SpatialLM (manycore-research, model weights CC-BY-NC-4.0) reads the scene's
# point cloud and returns walls, doors, windows and object boxes.
# Runs with the python of its own venv (SPATIALLM_DIR/venv): python run_spatiallm.py <scene_folder> [seed]
# Needs <scene_folder>\point_cloud.ply (export_point_cloud.py). Writes spatiallm.txt (raw) and spatiallm.json.
# flash-attn is not available on Windows, so the Sonata point encoder runs with plain attention.
import json, os, sys

ROOT = os.environ.get("SPATIALLM_DIR", "spatiallm")
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(os.path.join(ROOT, "code"))  # inference.py reads code_template.txt from here

import numpy as np
import torch
import spatiallm.model.sonata_encoder as sonata

_init = sonata.Sonata.__init__


def _init_no_flash(self, *a, **k):
    k["enable_flash"] = False
    _init(self, *a, **k)


sonata.Sonata.__init__ = _init_no_flash

from transformers import AutoTokenizer, AutoModelForCausalLM
from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd
from inference import preprocess_point_cloud, generate_layout

MODEL = os.path.join(ROOT, "SpatialLM1.1-Qwen-0.5B")


def main():
    folder = os.path.abspath(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    num_bins = model.config.point_config["num_bins"]
    grid = Layout.get_grid_size(num_bins)
    pcd = cleanup_pcd(load_o3d_pcd(os.path.join(folder, "point_cloud.ply")), voxel_size=grid)
    pts, cols = get_points_and_colors(pcd)
    lo = np.min(pts, axis=0)
    layout = generate_layout(model, preprocess_point_cloud(pts, cols, grid, num_bins), tok, "code_template.txt",
                             seed=seed, detect_type="all")
    layout.translate(lo)
    meta = os.path.join(folder, "point_cloud.json")
    turn = json.load(open(meta))["turn_deg"] if os.path.exists(meta) else 0.0
    layout.rotate(-np.radians(turn))  # back to the scene's floor frame
    open(os.path.join(folder, "spatiallm.txt"), "w").write(layout.to_language_string())
    out = dict(source="SpatialLM1.1-Qwen-0.5B", seed=seed, input_turn_deg=turn,
               walls=[dict(id=f"S{w.id}", start_m=[w.ax, w.ay], end_m=[w.bx, w.by], height_m=w.height, thickness_m=w.thickness)
                      for w in layout.walls],
               doors=[dict(wall=f"S{d.wall_id}", center_m=[d.position_x, d.position_y, d.position_z], width_m=d.width, height_m=d.height)
                      for d in layout.doors],
               windows=[dict(wall=f"S{d.wall_id}", center_m=[d.position_x, d.position_y, d.position_z], width_m=d.width, height_m=d.height)
                        for d in layout.windows],
               boxes=[dict(name=b.class_name, center_m=[b.position_x, b.position_y, b.position_z], angle_deg=float(np.degrees(b.angle_z)),
                           size_m=[b.scale_x, b.scale_y, b.scale_z]) for b in layout.bboxes])
    json.dump(out, open(os.path.join(folder, "spatiallm.json"), "w"), indent=1, default=float)
    print(f"SpatialLM: {len(out['walls'])} walls, {len(out['doors'])} doors, {len(out['windows'])} windows, {len(out['boxes'])} boxes")


if __name__ == "__main__":
    main()
