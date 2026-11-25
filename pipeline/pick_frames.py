# Picks the sharpest frame of every time window of one or more videos and writes them at full size and 960 wide.
# Usage: python -I pick_frames.py <scene_folder> <window_s> <video> [video ...]
#   -> <scene>/frames_full/<p>_<NNN>_t<sec>.png, <scene>/frames_960/ (same names), <scene>/video/frames.json
#   p = a, b, c... one letter per video, so names stay unique when a room is filmed in several clips.
import json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image

SCENE, WIN, VIDEOS = os.path.abspath(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
FPS = 4  # candidates per second
for d in ("frames_full", "frames_960", "video"):
    os.makedirs(os.path.join(SCENE, d), exist_ok=True)


def ffmpeg(args, tries=5):
    """ffmpeg 9.0.2 sometimes crashes (0xC0000005) on a command that works when run again, so retry."""
    for k in range(tries):
        if subprocess.run(["ffmpeg"] + args).returncode == 0: return
        print("ffmpeg failed, retry", k + 1, flush=True)
    raise RuntimeError("ffmpeg kept failing: " + " ".join(args))


def sharpness(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


picked = []
for k, vid in enumerate(VIDEOS):
    p = "abcdefghij"[k]
    with tempfile.TemporaryDirectory() as tmp:
        ffmpeg(["-loglevel", "error", "-y", "-i", vid, "-vf", f"fps={FPS},scale=540:-2", f"{tmp}/f_%04d.png"])
        files = sorted(os.listdir(tmp))
        cands = [((i + 0.5) / FPS, sharpness(Image.open(os.path.join(tmp, f)))) for i, f in enumerate(files)]
    buckets = {}
    for t, s in cands:
        b = int(t // WIN)
        if b not in buckets or s > buckets[b][1]:
            buckets[b] = (t, s)
    n = 0
    for b in sorted(buckets):
        t, s = buckets[b]
        name = f"{p}_{n:03d}_t{t:05.2f}"; n += 1
        full = os.path.join(SCENE, "frames_full", name + ".png")
        ffmpeg(["-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", vid, "-frames:v", "1", full])
        im = Image.open(full); im.resize((960, round(im.height * 960 / im.width)), Image.LANCZOS).save(os.path.join(SCENE, "frames_960", name + ".png"))
        picked.append(dict(name=name, video=os.path.abspath(vid), t=round(t, 2), sharpness=round(s, 1)))
    print(os.path.basename(vid), "->", n, "frames", flush=True)
json.dump(picked, open(os.path.join(SCENE, "video", "frames.json"), "w"), indent=1)
print("TOTAL", len(picked))
