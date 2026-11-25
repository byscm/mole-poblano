# DINOv3 visual fingerprint (the model already in your ComfyUI, nothing to install), used to find in the
# gallery the 3D model that looks most like an object cropped from a photo. Query part only: the gallery
# fingerprints (huellas.npz) are built by the Spanish scripts\huellas.py ("actualizar").
#
# Run with the ComfyUI .venv python (not -I):
#   ...\ComfyUI\ComfyUI\.venv\Scripts\python.exe
#
# From other scripts:  from fingerprints import Fingerprints, to_square
#                      fp = Fingerprints();  fp.search(to_square(rgb, mask), classes=None, k=10)

import os, sys
import numpy as np

COMFY = os.environ.get("COMFY_DIR", os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI"))
DINO = os.environ.get("DINO_WEIGHTS", os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Comfy-Desktop\ComfyUI-Shared\models\clip_vision\dino_v3_L_naf_fp32.safetensors"))
GALLERY = os.environ.get("GALLERY_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gallery"))
DATABASE = os.path.join(GALLERY, "huellas.npz")  # existing gallery data: Spanish file and key names
GRAY = 128
SIDE = 224

sys.path.insert(0, COMFY)


def to_square(rgb, mask=None):
    """Gray background outside the mask, tight crop and padding to a square."""
    from PIL import Image
    rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
    if mask is None:
        mask = np.ones(rgb.shape[:2], bool)
    img = np.where(mask[..., None], rgb, GRAY).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.full((SIDE, SIDE, 3), GRAY, np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    L = int(max(h, w) * 1.08) + 2
    canvas = np.full((L, L, 3), GRAY, np.uint8)
    canvas[(L - h) // 2:(L - h) // 2 + h, (L - w) // 2:(L - w) // 2 + w] = crop
    return np.array(Image.fromarray(canvas).resize((SIDE, SIDE), Image.BILINEAR))


class Fingerprints:
    def __init__(self):
        import torch
        import comfy.clip_vision, comfy.clip_model, comfy.model_management as mm
        self.torch, self.cm, self.mm = torch, comfy.clip_model, mm
        self.cv = comfy.clip_vision.load(DINO)
        self.library = None

    def vectors(self, images, batch=48):
        torch = self.torch
        self.mm.load_model_gpu(self.cv.patcher)
        out = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.stack(images[i:i + batch])).float() / 255.0
            pv = self.cm.clip_preprocess(x.to(self.cv.load_device), size=self.cv.image_size,
                                         mean=self.cv.image_mean, std=self.cv.image_std, crop=True).float()
            with torch.no_grad():
                r = self.cv.model(pixel_values=pv)
            cls = r[2].float()
            patches = r[0][:, 1:].float().mean(1)
            v = torch.cat([torch.nn.functional.normalize(cls, dim=-1), torch.nn.functional.normalize(patches, dim=-1)], -1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, 1), np.float16)

    def search(self, crop, classes=None, k=10):
        """crop: square RGB image (use to_square). classes: list of allowed gallery object names
        (e.g. ['cama_matrimonial', 'cama_individual']); None = whole gallery.
        Returns a list of (similarity, stem, yaw, elev), without repeating a model."""
        if self.library is None:
            d = np.load(DATABASE, allow_pickle=True)
            self.library = (d["emb"].astype(np.float32), d["tallo"], d["yaw"], d["elev"])
        E, T, Y, El = self.library
        q = self.vectors([crop])[0].astype(np.float32)
        s = E @ q
        if classes:
            ok = np.array([t.split("__v")[0] in classes for t in T])
            s = np.where(ok, s, -9)
        res, seen = [], set()
        for i in np.argsort(-s):
            if s[i] < -1 or T[i] in seen:
                continue
            seen.add(T[i])
            res.append((float(s[i]), str(T[i]), int(Y[i]), int(El[i])))
            if len(res) >= k:
                break
        return res
