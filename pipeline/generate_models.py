# Generates the 3D model (shape + texture) of every object in the map from its background-free crop,
# with TRELLIS 2 in your ComfyUI. The model comes out with arbitrary scale and yaw; build_scene.py
# scales it to the measured height and turns it to where it faces.
# Usage: python -I generate_models.py <map_folder> [port] [comma separated ids]
# Writes: <folder>/generated/object_N.glb and object_N_quality.json

import glob, json, os, shutil, sys, uuid
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import DEFAULT_PORT, upload_image, run_workflow, output_dir
from silhouette import fit_pose
from fit_poses import elevation

SEEDS = (42, 7, 1234)
GOOD = 0.78


def workflow(img, mask, prefix, seed=42):
    N = lambda c, **i: {"class_type": c, "inputs": i}
    return {
        "img": N("LoadImage", image=img),
        "msk": N("LoadImageMask", image=mask, channel="red"),
        "crop": N("ImageCropToMask", width=1024, height=1024, pad_factor=1.0, grow_mask=0, background="#000000",
                  images=["img", 0], masks=["msk", 0]),
        "dino": N("CLIPVisionLoader", clip_name="dino_v3_L_naf_fp32.safetensors"),
        "cond": N("Trellis2Conditioning", clip_vision_model=["dino", 0], image=["crop", 0]),
        "unet": N("UNETLoader", unet_name="trellis_2_int8_convrot.safetensors", weight_dtype="default"),
        "o199": N("CFGOverride", cfg=1, start_percent=0.667, end_percent=1, model=["unet", 0]),
        "o279": N("CFGOverride", cfg=1, start_percent=0.769, end_percent=1, model=["unet", 0]),
        "r125": N("RescaleCFG", multiplier=0.7, model=["o199", 0]),
        "r126": N("RescaleCFG", multiplier=0.5, model=["o279", 0]),
        "ms": N("ModelSamplingSD3", shift=5, model=["r125", 0]),
        "lat": N("EmptyTrellis2LatentStructure", batch_size=1),
        "vs": N("VAELoader", vae_name="trellis_2_shape_vae_bf16.safetensors"),
        "vt": N("VAELoader", vae_name="trellis_2_texture_vae_bf16.safetensors"),
        "k3": N("KSampler", seed=seed + 14, steps=12, cfg=7.5, sampler_name="euler", scheduler="normal", denoise=1,
                model=["ms", 0], positive=["cond", 0], negative=["cond", 1], latent_image=["lat", 0]),
        "struct": N("VaeDecodeStructureTrellis2", resolution="32", samples=["k3", 0], vae=["vs", 0]),
        "shape": N("Trellis2ShapeStage", positive=["cond", 0], negative=["cond", 1], voxel=["struct", 0]),
        "k18": N("KSampler", seed=seed, steps=20, cfg=7.5, sampler_name="euler", scheduler="normal", denoise=1,
                 model=["r126", 0], positive=["shape", 0], negative=["shape", 1], latent_image=["shape", 2]),
        "up": N("Trellis2UpsampleStage", target_resolution=1536, positive=["shape", 0], negative=["shape", 1],
                shape_latent=["k18", 0], vae=["vs", 0]),
        "k23": N("KSampler", seed=seed, steps=12, cfg=7.5, sampler_name="euler", scheduler="simple", denoise=1,
                 model=["r126", 0], positive=["up", 0], negative=["up", 1], latent_image=["up", 2]),
        "mesh": N("VaeDecodeShapeTrellis", samples=["k23", 0], vae=["vs", 0]),
        "tex": N("Trellis2TextureStage", positive=["up", 0], negative=["up", 1], shape_latent=["k23", 0]),
        "k12": N("KSampler", seed=seed + 1, steps=12, cfg=1, sampler_name="euler", scheduler="normal", denoise=1,
                 model=["unet", 0], positive=["tex", 0], negative=["tex", 1], latent_image=["tex", 2]),
        "colvox": N("VaeDecodeTextureTrellis", samples=["k12", 0], vae=["vt", 0], shape_subdivides=["mesh", 1]),
        "info": N("GetMeshInfo", mesh=["mesh", 0]),
        "remesh": N("RemeshMesh", resolution=768, sign_mode="solid", **{"sign_mode.fill": 20, "sign_mode.qef": False},
                    band=1, project_back=0, fix_poles=False, smooth_iters=20, drop_small_components=0.01,
                    precluster_max_verts=20000000, mesh=["info", 0]),
        "dec": N("DecimateMesh", target_face_count=150000, placement_mode="midpoint", mesh=["remesh", 0]),
        "smooth": N("MeshSmoothNormals", crease_angle=180, mesh=["dec", 0]),
        "paint": N("PaintMesh", mesh=["smooth", 0], voxel_colors=["colvox", 0]),
        "save": N("SaveGLB", mesh=["paint", 0], filename_prefix=prefix),
    }


def generate_one(img_name, mask_name, prefix, seed, port):
    """Runs TRELLIS once (waits up to 30 min) and returns the newest GLB it saved."""
    run_workflow(workflow(img_name, mask_name, prefix, seed), port=port, timeout=1800)
    return sorted(glob.glob(os.path.join(output_dir(port), prefix + "*.glb")), key=os.path.getmtime)[-1]


def main():
    scene = sys.argv[1]
    port = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PORT
    only = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
    scene_map = json.load(open(os.path.join(scene, "map.json"), encoding="utf-8"))
    gen = os.path.join(scene, "generated")
    os.makedirs(gen, exist_ok=True)
    for o in scene_map["objects"]:
        if only and o["id"] not in only:
            continue
        if o.get("model"):  # the gallery already has a very similar one
            continue
        dst = os.path.join(gen, f"object_{o['id']}.glb")
        if os.path.exists(dst):
            continue
        rgba = np.array(Image.open(os.path.join(scene, o["crop"])).convert("RGBA"))
        tag = uuid.uuid4().hex[:6]
        p_img, p_msk = os.path.join(gen, f"_in_{o['id']}.png"), os.path.join(gen, f"_m_{o['id']}.png")
        Image.fromarray(rgba[..., :3]).save(p_img)
        Image.fromarray(rgba[..., 3]).save(p_msk)
        n_img, n_msk = upload_image(p_img, port), upload_image(p_msk, port)
        attempts = []
        for seed in SEEDS:
            glb = generate_one(n_img, n_msk,
                               f"generated/{os.path.basename(os.path.normpath(scene))}_obj{o['id']}_{tag}_s{seed}",
                               seed, port)
            # same check as the final orientation: seen from the real camera and without the TRELLIS fake slab
            q = fit_pose(glb, os.path.join(scene, o["crop"]), elevation(scene_map, o))
            score = q["iou"] - (0.5 if q["relative_thickness"] < 0.02 else 0)  # a flat plate is a failure
            attempts.append((score, seed, glb, q))
            print(f"object {o['id']} ({o['name']}) seed {seed}: silhouette {q['iou']:.2f}  "
                  f"thickness {q['relative_thickness']:.3f}", flush=True)
            if score >= GOOD:
                break
        score, seed, glb, q = max(attempts, key=lambda t: t[0])
        shutil.copy(glb, dst)
        json.dump(dict(seed=seed, **q, attempts=[(t[1], t[3]) for t in attempts]),
                  open(dst.replace(".glb", "_quality.json"), "w"), indent=1)
        for p in (p_img, p_msk):  # temporary inputs written above by this script
            os.remove(p)
        print(f"  -> keeping seed {seed} (silhouette {q['iou']:.2f})"
              f"{'' if score >= GOOD else '  ** none good, review **'}", flush=True)


if __name__ == "__main__":
    main()
