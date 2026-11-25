# Hi3D v1.0 (SPARC3D) refinement of a coarse TRELLIS mesh: the mesh + its reference image go in, a more detailed mesh
# (geometry only, no colour) comes out. Passes chain upwards, e.g. 512,1024 or 1024 alone (each decoded mesh feeds the next).
# Needs the Aero-Ex-ComfyUI-Hi3D node and hi3d_v1_int8 + dinov2-large + clip-vit-large-patch14 (ComfyUI restarted after install).
# Usage: python -I refine_hi3d.py <reference.png> <coarse.glb> <out.glb> [port=8188] [passes=1024] [seed=0]
import json, os, shutil, sys, time, uuid, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import upload_image, output_dir

ref, coarse, out = sys.argv[1:4]
port = int(sys.argv[4]) if len(sys.argv) > 4 else 8188
passes = (sys.argv[5] if len(sys.argv) > 5 else "1024").split(",")
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
INPUT = os.environ.get("COMFY_INPUT_DIR", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Comfy-Desktop", "ComfyUI-Shared", "input"))
tag = uuid.uuid4().hex[:8]; mesh_name = f"3d/refine_{tag}.glb"
os.makedirs(os.path.join(INPUT, "3d"), exist_ok=True); shutil.copy(coarse, os.path.join(INPUT, mesh_name))
img = upload_image(ref, port)

wf = {
    "1": {"class_type": "Hi3DCheckpointLoader", "inputs": {"ckpt_name": "hi3d_v1_int8.safetensors"}},
    "2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dinov2-large.safetensors"}},
    "3": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip-vit-large-patch14.safetensors"}},
    "4": {"class_type": "LoadImage", "inputs": {"image": img}},
    "5": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
    "6": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["5", 0], "image": ["4", 0]}},
    "7": {"class_type": "ImageCropToMask", "inputs": {"images": ["4", 0], "masks": ["6", 0], "width": 1024, "height": 1024,
                                                      "pad_factor": 1.2, "grow_mask": 0, "background": "#ffffff"}},
    "9": {"class_type": "Hi3DConditioning", "inputs": {"dino_vision": ["2", 0], "clip_vision": ["3", 0], "image": ["7", 0]}},
    "16": {"class_type": "Load3DAdvanced", "inputs": {"model_file": mesh_name, "viewport_state": {}, "width": 1024, "height": 1024}},
    "17": {"class_type": "Get3DComponents", "inputs": {"model_3d": ["16", 0]}},
    "18": {"class_type": "RemeshMesh", "inputs": {"mesh": ["17", 0], "resolution": 1024, "sign_mode": "solid", "sign_mode.fill": 20.0,
                                                  "sign_mode.qef": False, "band": 1.0, "project_back": 0.0, "fix_poles": False,
                                                  "smooth_iters": 0, "drop_small_components": 0.01, "precluster_max_verts": 20000000}},
}
mesh = ["18", 0]
for k, res in enumerate(passes):
    s, a, d = f"s{k}", f"a{k}", f"d{k}"
    wf[s] = {"class_type": "Hi3DRefineStage", "inputs": {"positive": ["9", 0], "negative": ["9", 1], "mesh": mesh, "resolution": res}}
    wf[a] = {"class_type": "Hi3DSampler", "inputs": {"model": ["1", 0], "positive": [s, 0], "negative": [s, 1], "latent": [s, 2],
                                                      "seed": seed, "steps": 25}}
    wf[d] = {"class_type": "VAEDecodeHi3D", "inputs": {"samples": [a, 0], "vae": ["1", 1]}}
    mesh = [d, 0]
wf["save"] = {"class_type": "SaveGLB", "inputs": {"mesh": mesh, "filename_prefix": f"hi3d/{tag}"}}
wf["crop_preview"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"hi3d/{tag}_ref"}}

url = f"http://127.0.0.1:{port}"
req = lambda p, body=None: json.loads(urllib.request.urlopen(urllib.request.Request(
    url + p, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"}), timeout=60).read())
t = time.time()
try:
    pid = req("/prompt", {"prompt": wf, "client_id": tag})["prompt_id"]
except urllib.error.HTTPError as e:
    raise SystemExit("rejected: " + e.read().decode()[:2000])
while True:
    h = req("/history/" + pid)
    if pid in h and h[pid].get("status", {}).get("completed") is not None: break
    time.sleep(3)
st = h[pid]["status"]
if st.get("status_str") != "success": raise SystemExit("failed: " + json.dumps(st)[:2000])
outs = h[pid]["outputs"]; base = output_dir(port)
files = [f for o in outs.values() for key in ("3d", "images", "result") for f in (o.get(key) or []) if isinstance(f, dict)]
glb = next(f for f in files if f["filename"].endswith(".glb")); refimg = next((f for f in files if f["filename"].endswith(".png")), None)
shutil.copy(os.path.join(base, glb.get("subfolder", ""), glb["filename"]), out)
if refimg: shutil.copy(os.path.join(base, refimg.get("subfolder", ""), refimg["filename"]), out[:-4] + "_ref.png")
print(f"OK {out} passes={','.join(passes)} {time.time() - t:.0f} s")
