# One clean, whole view of an object from SEVERAL partial views of it (different photos / sides), made by
# Qwen-Image 2.1 with the views as reference images, on a flat solid background that is easy to cut out.
# The result is meant as TRELLIS input (model_from_view.py): a whole object, front three-quarter, no clutter.
# Usage: python -I clean_view.py <job.json> [port]
#   job: {"out": folder, "name": str, "views": [{"scene": path, "id": int, "what": "back three-quarter view"}...],
#         "object": "a dining chair ...", "view": "front three-quarter ...", "background": [0, 200, 0],
#         "seeds": [1, 2, 3]}
# Writes <out>/<name>_refs.png (what the model saw), <out>/<name>_s<seed>.png
import json, os, sys, time, uuid, urllib.parse, urllib.request
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_client import upload_image

PAD = 0.12  # context around the object's box


def view_crop(scene, oid=None, box=None):
    """Context crop of the photo around a detected object (oid) or around a box given by hand (photo pixels)."""
    m = json.load(open(os.path.join(scene, "map.json"), encoding="utf-8"))
    im = Image.open(os.path.join(scene, "photo.png")).convert("RGB")
    if box is not None:
        return im.crop(tuple(int(v) for v in box))
    o = next(x for x in m["objects"] if x["id"] == oid)
    sx, sy = im.width / m["W"], im.height / m["H"]
    x0, y0, x1, y1 = o["image_box"]
    w, h = x1 - x0, y1 - y0
    box = (max(0, (x0 - PAD * w) * sx), max(0, (y0 - PAD * h) * sy), min(im.width, (x1 + PAD * w) * sx), min(im.height, (y1 + PAD * h) * sy))
    return im.crop(tuple(int(v) for v in box))


def workflow(prompt, neg, refs, seed, w, h):
    enc = {"clip": ["3", 0], "prompt": prompt, "negative_prompt": neg, "resolution": 768, "vae": ["4", 0]}
    f = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
         "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
         "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
         "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
         "5": {"class_type": "TextEncodeQwenImage21", "inputs": enc},
         "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
         "7": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["6", 0],
                                                    "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
         "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["4", 0]}},
         "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "clean_view"}}}
    for k, name in enumerate(refs, 1):
        f[f"r{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{k}"] = [f"r{k}", 0]
    return f


def run(url, wf):
    req = lambda p, d=None: json.loads(urllib.request.urlopen(urllib.request.Request(
        url + p, data=json.dumps(d).encode() if d else None, headers={"Content-Type": "application/json"}), timeout=60).read())
    pid = req("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = req("/history/" + pid)
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            break
        time.sleep(2)
    if h[pid]["status"].get("status_str") != "success":
        raise RuntimeError(json.dumps(h[pid]["status"])[:800])
    img = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])][0]
    return urllib.request.urlopen(f"{url}/view?filename={urllib.parse.quote(img['filename'])}"
                                  f"&subfolder={urllib.parse.quote(img['subfolder'])}&type=output").read()


def main():
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    port = sys.argv[2] if len(sys.argv) > 2 else "8188"
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    for job in jobs if isinstance(jobs, list) else [jobs]:
        if only is None or job["name"] in only:
            one(job, port)


def one(job, port):
    url = f"http://127.0.0.1:{port}"
    os.makedirs(job["out"], exist_ok=True)
    crops = [view_crop(v["scene"], v.get("id"), v.get("box")) for v in job["views"]]
    names = []
    for k, c in enumerate(crops, 1):
        p = os.path.join(job["out"], f"{job['name']}_ref{k}.png"); c.save(p); names.append(upload_image(p, port))
    # sheet of what the model was given
    th = [c.copy() for c in crops]
    for t in th: t.thumbnail((300, 300))
    sheet = Image.new("RGB", (310 * len(th), 340), "white"); d = ImageDraw.Draw(sheet)
    for k, (t, v) in enumerate(zip(th, job["views"])):
        sheet.paste(t, (310 * k + 5, 5)); d.text((310 * k + 8, 312), f"image {k+1}: {v['what']}", fill="black")
    sheet.save(os.path.join(job["out"], f"{job['name']}_refs.png"))

    bg = job.get("background", [0, 200, 0])
    lines = [f"Image {k}: {v['what']}." for k, v in enumerate(job["views"], 1)]
    prompt = (f"The {len(crops)} reference images are photos of the SAME object, {job['object']}, taken from different sides, "
              f"sometimes partly hidden by other things. " + " ".join(lines) +
              f" Using all of them together, create one clean product photo of exactly this object: {job['view']}. "
              f"The whole object is visible and centered with a margin, nothing cut off, nothing else in the picture. "
              f"Flat solid pure background color RGB({bg[0]},{bg[1]},{bg[2]}) everywhere, no floor, no shadow, no gradient. "
              f"Keep its real materials, colors, proportions and details exactly as in the photos. {job.get('extra', '')} "
              f"Soft even studio light, sharp focus.")
    neg = "other furniture, table, people, room, floor, shadow, gradient background, text, watermark, cropped, blurry, extra legs, deformed"
    open(os.path.join(job["out"], f"{job['name']}_prompt.txt"), "w", encoding="utf-8").write(prompt + "\n\nNEGATIVE: " + neg)
    for seed in job.get("seeds", [1, 2, 3]):
        dest = os.path.join(job["out"], f"{job['name']}_s{seed}.png")
        if os.path.exists(dest):
            continue
        t0 = time.time()
        open(dest, "wb").write(run(url, workflow(prompt, neg, names, seed, job.get("w", 1024), job.get("h", 1024))))
        print(f"{os.path.basename(dest)}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
