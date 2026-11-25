# ComfyUI HTTP client shared by the photo pipeline.
# Port of the HTTP/workflow parts of scripts\sam_escena.py, scripts\medir_escena.py and scripts\vlm.py.
#   upload_image  - upload a PNG for LoadImage
#   run_workflow  - queue a workflow and wait for its history entry
#   sam_masks     - SAM 3 text-prompted masks (one per detected instance)
#   moge_points   - MoGe-2 point map (one 3D point per pixel, GLB coordinates)
#   ask_vlm       - ask Qwen3-VL (TextGenerate node) a question about an image
# Command line: python -I comfy_client.py <image> "<question>" [port]

import glob, json, os, struct, sys, time, uuid, urllib.error, urllib.request
import numpy as np
from PIL import Image

DEFAULT_PORT = "8188"
# ComfyUI output folder per port, e.g. COMFY_OUTPUT_DIRS='{"8188": "D:/comfy/out1", "8190": "D:/comfy/out2"}'.
# Without it, every port uses COMFY_OUTPUT_DIR (default: the ComfyUI Desktop output folder).
OUTPUT_DIRS = json.loads(os.environ.get("COMFY_OUTPUT_DIRS", "{}"))
DEFAULT_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Comfy-Desktop", "ComfyUI-Shared", "output"))

SAM_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
SAM_THRESHOLD = 0.4
MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
VLM_MODEL = "qwen3vl_8b_int8_convrot.safetensors"
VLM_TYPE = "qwen_image"

POLL_SECONDS = 0.5
# Timeouts the Spanish code effectively used (polls x sleep)
SAM_TIMEOUT = 150
MOGE_TIMEOUT = 600
VLM_TIMEOUT = 450


def base_url(port):
    return f"http://127.0.0.1:{port}"


def output_dir(port):
    return OUTPUT_DIRS.get(str(port), DEFAULT_OUTPUT_DIR)


def upload_image(path, port=DEFAULT_PORT):
    """Upload a PNG to ComfyUI; returns the name to use in LoadImage."""
    name = f"upload_{uuid.uuid4().hex[:8]}.png"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base_url(port) + "/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))["name"]


def run_workflow(workflow, port=DEFAULT_PORT, timeout=1800):
    """Queue a workflow (API format) and wait for it. Returns the history entry
    (dict with "outputs" and "status"). Raises RuntimeError on error, TimeoutError on timeout."""
    url = base_url(port)
    req = urllib.request.Request(url + "/prompt", json.dumps({"prompt": workflow}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()[:1500])
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"{url}/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st.get("messages", ""))[-1500:])
            if st.get("completed"):
                return h[pid]
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"workflow {pid} did not finish in {timeout} s")


def sam_masks(image_name, phrase, subfolder, port=DEFAULT_PORT, max_det=1):
    """SAM 3 masks for a text phrase. Returns (list of bool arrays at image resolution, error or None)."""
    # SAM 3 in ComfyUI returns 1 detection per phrase unless the text is "phrase:N"
    text = phrase if max_det <= 1 else f"{phrase}:{max_det}"
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CHECKPOINT}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 1]}},
        "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "conditioning": ["3", 0],
                                                       "threshold": SAM_THRESHOLD, "refine_iterations": 2,
                                                       "individual_masks": True}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0],
                                                     "filename_prefix": f"sam/{subfolder}/{phrase.replace(' ', '_')}"}},
    }
    try:
        entry = run_workflow(workflow, port, timeout=SAM_TIMEOUT)
    except (RuntimeError, TimeoutError) as e:
        return [], str(e)[-600:]
    masks = []
    for im in entry["outputs"].get("6", {}).get("images", []):
        path = os.path.join(output_dir(port), im.get("subfolder", ""), im["filename"])
        masks.append(np.array(Image.open(path).convert("L")) > 127)
    return masks, None


def _read_glb_positions(path):
    """POSITION attribute of the first mesh primitive of a GLB, as (N, 3) float32."""
    b = open(path, "rb").read()
    n_json = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(b[20:20 + n_json])
    bin0 = 20 + n_json + 8
    prim = js["meshes"][0]["primitives"][0]
    acc = js["accessors"][prim["attributes"]["POSITION"]]
    bv = js["bufferViews"][acc["bufferView"]]
    off = bin0 + bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(b, np.float32, acc["count"] * 3, off).reshape(-1, 3).copy()


def moge_points(image_name, subfolder, port=DEFAULT_PORT, fov_deg=0.0):
    """MoGe-2 point map of an uploaded image: (H*W, 3) float32 in GLB axes (Y up, camera looks to -Z).
    fov_deg = horizontal field of view; 0 lets MoGe guess it."""
    prefix = f"moge/{subfolder}"
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadMoGeModel", "inputs": {"model_name": MOGE_MODEL}},
        "3": {"class_type": "MoGeInference", "inputs": {"moge_model": ["2", 0], "image": ["1", 0], "resolution_level": 9,
                                                         "fov_x_degrees": float(fov_deg or 0.0), "batch_size": 1,
                                                         "force_projection": True, "apply_mask": False, "refine_steps": 0}},
        "4": {"class_type": "MoGePointMapToMesh", "inputs": {"moge_geometry": ["3", 0], "batch_index": 0, "decimation": 1,
                                                              "discontinuity_threshold": 0.0, "texture": False}},
        "5": {"class_type": "SaveGLB", "inputs": {"mesh": ["4", 0], "filename_prefix": prefix}},
    }
    run_workflow(workflow, port, timeout=MOGE_TIMEOUT)
    files = sorted(glob.glob(os.path.join(output_dir(port), prefix + "*.glb")), key=os.path.getmtime)
    if not files:
        raise RuntimeError(f"MoGe finished but no GLB found for prefix {prefix}")
    return _read_glb_positions(files[-1])


def ask_vlm(image_path, question, port=DEFAULT_PORT, max_len=400):
    """Ask the local vision-language model (Qwen3-VL) a question about an image; returns its text answer."""
    image_name = upload_image(image_path, port)
    prefix = f"vlm/{uuid.uuid4().hex[:8]}"
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": VLM_MODEL, "type": VLM_TYPE}},
        "3": {"class_type": "TextGenerate", "inputs": {"clip": ["2", 0], "prompt": question, "max_length": max_len,
                                                        "sampling_mode": "off", "image": ["1", 0]}},
        "4": {"class_type": "SaveText", "inputs": {"text": ["3", 0], "filename_prefix": prefix, "format": "txt"}},
    }
    out = run_workflow(workflow, port, timeout=VLM_TIMEOUT)["outputs"]
    # the text can come in the UI output or in a file
    for v in out.values():
        for field in ("text", "string"):
            if field in v:
                t = v[field]
                return t[0] if isinstance(t, list) else t
    for v in out.values():
        for fi in v.get("text_files", []) + v.get("files", []) + v.get("texts", []):
            if isinstance(fi, dict) and "filename" in fi:
                return open(os.path.join(output_dir(port), fi.get("subfolder", ""), fi["filename"]), encoding="utf-8").read()
    files = sorted(glob.glob(os.path.join(output_dir(port), prefix + "*")), key=os.path.getmtime)
    if files:
        return open(files[-1], encoding="utf-8").read()
    return json.dumps(out)[:2000]


if __name__ == "__main__":
    a = sys.argv[1:]
    print(ask_vlm(a[0], a[1], *(a[2:3])))
