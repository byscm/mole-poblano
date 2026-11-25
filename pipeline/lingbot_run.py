# LingBot-Map runner for the evaluation (no viewer). Loads weights from .safetensors only (never the pickle .pt).
# Run with the python of your lingbot-map venv; LINGBOT_DIR points to the cloned repo.
#   run_lingbot.py --weights W.safetensors --out out_folder (--images folder | --video file.mp4 --fps 5) [--mode streaming|windowed] [--sdpa]
# Writes out_folder/pred.npz (extrinsic w2c 3x4, intrinsic, depth f16, depth_conf f16, frame paths) and run.json (time, VRAM).
import argparse, json, os, sys, time
REPO = os.environ.get("LINGBOT_DIR", "lingbot-map")
sys.path.insert(0, REPO)
import numpy as np
import torch
from safetensors.torch import load_file
import demo  # repo's demo.py: load_images, load_model, postprocess

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--out", required=True)
p.add_argument("--images"); p.add_argument("--video"); p.add_argument("--fps", type=int, default=5)
p.add_argument("--mode", default="streaming", choices=["streaming", "windowed"])
p.add_argument("--sdpa", action="store_true")
p.add_argument("--num_scale_frames", type=int, default=8)
p.add_argument("--keyframe_interval", type=int, default=None)
p.add_argument("--window_size", type=int, default=64)
p.add_argument("--overlap_size", type=int, default=16)
p.add_argument("--save_points", action="store_true", help="also save world_points (f16, every 2nd pixel)")
p.add_argument("--no_release", action="store_true", help="do not empty the CUDA cache after every frame")
p.add_argument("--no_offload", action="store_true", help="keep per-frame outputs on the GPU (default: offload to CPU)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")

t_all = time.time()
images, paths, _ = demo.load_images(image_folder=a.images, video_path=a.video, fps=a.fps)
n = images.shape[0]
margs = argparse.Namespace(mode=a.mode, image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                           num_scale_frames=min(a.num_scale_frames, max(1, n - 1)), kv_cache_sliding_window=64,
                           use_sdpa=a.sdpa, camera_num_iterations=4, model_path=None)
t0 = time.time()
model = demo.load_model(margs, torch.device("cpu"))
sd = load_file(a.weights)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"weights: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
model = model.to(dev).eval()
dtype = torch.bfloat16
model.aggregator = model.aggregator.to(dtype=dtype)
t_load = time.time() - t0
images = images.to(dev)
odev = None if a.no_offload else torch.device("cpu")
kfi = a.keyframe_interval or (1 if n <= 320 else (n + 319) // 320)
# Windows has no expandable_segments: the KV cache grows by torch.cat every frame and layer, the freed smaller blocks
# are never reused and reserved memory balloons (68 GB reserved for 10.7 GB used at 63 frames -> sysmem fallback, OOM at 64).
# Releasing the allocator cache after every frame keeps reserved close to allocated.
_fwd = model.forward
def _forward_then_release(*args, **kwargs):
    r = _fwd(*args, **kwargs)
    torch.cuda.empty_cache()
    return r
if not a.no_release:
    model.forward = _forward_then_release
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    if a.mode == "streaming":
        pred = model.inference_streaming(images, num_scale_frames=margs.num_scale_frames, keyframe_interval=kfi, output_device=odev)
    else:
        pred = model.inference_windowed(images, window_size=a.window_size, overlap_size=a.overlap_size, overlap_keyframes=None,
                                        num_scale_frames=margs.num_scale_frames, keyframe_interval=kfi, output_device=odev)
torch.cuda.synchronize()
t_inf = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9; peak_res = torch.cuda.max_memory_reserved() / 1e9
if odev is not None:
    del images; torch.cuda.empty_cache(); images = pred["images"]
pred, imgs_cpu = demo.postprocess(pred, images)
out = dict(extrinsic=pred["extrinsic"].float().numpy(), intrinsic=pred["intrinsic"].float().numpy(),
           depth=pred["depth"].float().numpy().astype(np.float16), depth_conf=pred["depth_conf"].float().numpy().astype(np.float16),
           paths=np.array(paths), image_hw=np.array(images.shape[-2:]))
if a.save_points and "world_points" in pred:
    out["world_points"] = pred["world_points"].float().numpy()[:, ::2, ::2].astype(np.float16)
    out["world_points_conf"] = pred["world_points_conf"].float().numpy()[:, ::2, ::2].astype(np.float16)
    out["colors"] = (imgs_cpu.float().numpy()[:, :, ::2, ::2].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
np.savez_compressed(os.path.join(a.out, "pred.npz"), **out)
info = dict(frames=int(n), image_hw=list(map(int, images.shape[-2:])), mode=a.mode, sdpa=a.sdpa, keyframe_interval=kfi,
            load_s=round(t_load, 1), inference_s=round(t_inf, 1), fps=round(n / t_inf, 1), peak_vram_gb=round(peak, 2), peak_reserved_gb=round(peak_res, 2),
            total_s=round(time.time() - t_all, 1), missing_keys=len(missing), unexpected_keys=len(unexpected),
            weights=a.weights, keys=list(pred.keys()))
json.dump(info, open(os.path.join(a.out, "run.json"), "w"), indent=1)
print(json.dumps(info))
