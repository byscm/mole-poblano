# MAP OF ANY PHOTO, even when the objects are not in the gallery.
# Qwen3-VL lists the kinds of things -> SAM 3 separates each one -> Qwen3-VL names each crop
# -> MoGe-2 gives 3D points -> position, height, width, depth and facing of each object, plus distances.
# Usage: python -I detect_objects.py <photo> <out_folder> [port] [--fov DEG]
#        python -I detect_objects.py --redraw <out_folder>     (redraw guides.png from an existing map)
# Writes: photo.png, map.json, object_masks.npz, moge_points.npz,
#         guides.png (photo with numbers, outlines and distance lines),
#         crops/object_N.png (transparent background, for 3D generation)

import json, math, os, re, sys
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comfy_client import DEFAULT_PORT, upload_image, sam_masks, moge_points, ask_vlm

MIN_AREA = 0.0015   # a crop on its own must cover at least 0.15 % of the image
MIN_SIBLING_PX = 60 # smaller crops are kept (down to 60 px at half resolution) when a bigger one of the same
                    # kind was already accepted: far or half-hidden chairs around a table, etc.
MAX_OBJECTS = 15    # at most the 15 largest
ONE_THING = re.compile(r"\b(stacks?|piles?|pallets?|sets?|groups?|rows?|bunch|heaps?|shelf|shelves|rack)\b", re.I)
PERSON = re.compile(r"\b(person|people|man|men|woman|women|worker|boy|girl|child|kid)\b", re.I)
# parts of the building, not objects (the VLM sometimes lists them anyway)
NOT_OBJECTS = re.compile(r"\b(windows?|window frames?|doors?|walls?|ceilings?|floors?|roofs?|blinds?|curtains?|"
                         r"lamps?|lights?|pendant|chandeliers?|beams?|columns?|railings?|stairs?)\b", re.I)


def exif_fov(img, W, H):
    try:
        ex = img.getexif()
        f35 = ex.get_ifd(0x8769).get(0xA405) or ex.get(0xA405)
        if not f35:
            return None
        diag = 2 * math.atan(43.27 / (2 * float(f35)))
        f_px = math.hypot(W, H) / 2 / math.tan(diag / 2)
        return math.degrees(2 * math.atan(W / 2 / f_px))
    except Exception:
        return None


def robust_floor(p):
    """Floor plane robust to grass and bumps: SVD + drop the outliers, several times."""
    p = p[np.isfinite(p).all(1)]
    for _ in range(4):
        c = np.median(p, 0)
        n = np.linalg.svd(p - c, full_matrices=False)[2][2]
        r = np.abs((p - c) @ n)
        p = p[r < max(np.percentile(r, 80), 0.01)]
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][2]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, d


def choose_floor(cands, obj_union, Pg):
    """Keep only floor masks that are really below the camera: most pixels in the lower half of the image
    and 3D points below the optical center (OpenCV y points down). Giant masks (>60 % of the image) are out.
    If nothing survives, fall back to the bottom fifth of the image that is not an object."""
    GH, GW = obj_union.shape
    lower = np.zeros((GH, GW), bool); lower[GH // 2:] = True
    floor = np.zeros((GH, GW), bool)
    for ph, m in cands:
        m = m & ~obj_union
        if m.sum() < 50 or m.mean() > 0.6:
            continue
        y = Pg[m][:, 1]
        y = y[np.isfinite(y)]
        if (m & lower).sum() / m.sum() > 0.6 and len(y) and np.median(y) > 0:
            floor |= m
        else:
            print(f"  floor candidate '{ph}' rejected (not below the camera)", flush=True)
    if floor.sum() < 50:
        floor = np.zeros((GH, GW), bool); floor[int(GH * 0.8):] = True
        floor &= ~obj_union
        print("  no floor mask accepted: using the bottom of the image", flush=True)
    return floor


def erode(m, k=1):
    e = m.copy()
    for _ in range(k):
        e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e.copy()
    return e


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    args = sys.argv[1:]
    if args and args[0] == "--redraw":  # redraw the guides from an existing map
        out = args[1]
        scene_map = json.load(open(os.path.join(out, "map.json"), encoding="utf-8"))
        masks = np.load(os.path.join(out, "object_masks.npz"))["masks"]
        rgb = np.array(Image.open(os.path.join(out, "photo.png")).convert("RGB"))
        draw(out, rgb, [dict(m=m) for m in masks], scene_map["objects"], scene_map["distances"])
        return
    fov = None
    if "--fov" in args:
        k = args.index("--fov"); fov = float(args[k + 1]); del args[k:k + 2]
    photo, out = args[0], args[1]
    port = args[2] if len(args) > 2 else DEFAULT_PORT
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    os.makedirs(os.path.join(out, "_vlm_inputs"), exist_ok=True)
    scene = os.path.basename(out)

    orig = Image.open(photo)
    img = ImageOps.exif_transpose(orig).convert("RGB")
    sc = min(1.0, 1280 / max(img.width, img.height))
    W, H = int(img.width * sc) // 4 * 4, int(img.height * sc) // 4 * 4
    fov_source = "given" if fov else None
    if fov is None:
        fov = exif_fov(orig, img.width, img.height)
        fov_source = "EXIF" if fov else "guessed by MoGe"
    img = img.resize((W, H), Image.LANCZOS)
    photo_path = os.path.join(out, "photo.png")
    img.save(photo_path)
    rgb = np.array(img)
    GH, GW = H // 2, W // 2

    def downsample(m):
        return m.reshape(GH, H // GH, GW, W // GW).mean((1, 3)) > 0.5

    # 1) what kinds of things are there
    listing = ask_vlm(photo_path, "List the main physical objects in this photo that stand on the floor or ground, "
                                  "or on furniture (not walls, windows, doors, floor, grass, sky, ceiling, lamps on walls). "
                                  "Treat a stack, pile or pallet of things as ONE object (e.g. 'stack of boxes', 'pallet of boxes'). "
                                  "Include people and vehicles. Skip small items under 20 cm. At most 10 lines. "
                                  "One per line, short English noun phrase (1-3 words), numbered. Group identical objects in one line.", port)
    phrases = []
    for l in listing.splitlines():
        l = re.sub(r"^\s*\d+[\.\)]\s*", "", l).strip().strip(".").lower()
        if l and len(l) < 40 and l not in phrases and not NOT_OBJECTS.search(l):
            phrases.append(l)
    print("object types:", phrases, flush=True)

    # 2) SAM 3 crops (each instance separately)
    image_name = upload_image(photo_path, port)
    cand = []
    for ph in phrases:
        ms, err = sam_masks(image_name, ph, scene, port, max_det=15)
        for m in ms:
            b = downsample(m)
            if b.sum() >= MIN_SIBLING_PX:  # tiny pieces: out (the size rule is applied below)
                cand.append([b, ph])
        print(f"  '{ph}': {len(ms)} masks {err or ''}", flush=True)
    cand.sort(key=lambda c: -c[0].sum())
    # a mask that just groups several instances of the same kind (e.g. "dining chairs" covering all of them):
    # drop it so the single ones survive
    groups = []
    for a, (A, pa) in enumerate(cand):
        if ONE_THING.search(pa):  # "stack of boxes", "pallet of ...", "sofa set": one object on purpose
            continue
        parts =[B for B, pb in cand[a + 1:] if pb == pa and (A & B).sum() > 0.8 * B.sum()]
        if len(parts) >= 2 and sum(B.sum() for B in parts) > 0.5 * A.sum():
            groups.append(a)
            print(f"  '{pa}': one mask groups {len(parts)} others -> using the single ones", flush=True)
    cand = [c for a, c in enumerate(cand) if a not in groups]
    objects = []
    alone = max(120, MIN_AREA * GH * GW)
    confirmed = set()  # phrases with at least one big accepted instance
    for b, ph in cand:
        if b.sum() < alone and ph not in confirmed:
            continue
        duplicate = False
        for o in objects:
            inter = (b & o["m"]).sum()
            if inter / b.sum() > 0.6:  # almost entirely inside a larger one: part of it (or the same one)
                if ph not in o["phrases"]:
                    o["phrases"].append(ph)
                duplicate = True
                break
        if not duplicate and len(objects) < MAX_OBJECTS:  # sorted large to small: the largest are kept
            objects.append(dict(m=b, phrases=[ph]))
            if b.sum() >= alone:
                confirmed.add(ph)
    # floor candidates (checked in 3D below: SAM sometimes returns walls, ceilings or half the image)
    floor_cands = []
    for ph in ("floor", "ground", "grass", "lawn"):
        ms, _ = sam_masks(image_name, ph, scene, port, max_det=5)
        floor_cands += [(ph, downsample(m)) for m in ms]
    obj_union = np.zeros((GH, GW), bool)
    for o in objects:
        obj_union |= o["m"]

    # 3) 3D points
    cache = os.path.join(out, "moge_points.npz")
    if not os.path.exists(cache):
        pts = moge_points(upload_image(photo_path, port), scene, port, fov_deg=fov or 0.0)
        np.savez_compressed(cache, points=pts)
    P = np.load(cache)["points"].reshape(H, W, 3)
    if np.nanmedian(P[..., 2]) < 0:  # GLB: Y up, camera looks to -Z -> OpenCV (x right, y down, z forward)
        P = P * np.array([1, -1, -1], np.float32)
    Pg = P.reshape(GH, 2, GW, 2, 3).mean((1, 3))
    uu = np.broadcast_to(np.arange(GW) + 0.5 - GW / 2, (GH, GW))
    fx = float(np.nansum(uu ** 2) / np.nansum(uu * Pg[..., 0] / Pg[..., 2]))
    fov_used = math.degrees(2 * math.atan(GW / 2 / fx))
    floor = choose_floor(floor_cands, obj_union, Pg)
    print(f"{len(objects)} separate objects; floor {floor.mean():.0%} of the image", flush=True)
    n, d = robust_floor(Pg[floor])
    if not 0.2 < d < 6.0:
        print(f"WARNING: camera height {d:.2f} m looks wrong; the floor was probably not found", flush=True)
    f = np.array([0, 0, 1.0]) - n[2] * n
    ey = f / np.linalg.norm(f)
    ex = np.cross(ey, n)
    Mx = np.stack([ex, ey, n])
    O = -d * n
    to_floor = lambda p: (p - O) @ Mx.T

    # 4) each object: name, position, size, facing
    res = []
    footprints = []  # (xy points on the floor, top, base) per object
    for i, o in enumerate(objects, 1):
        m = o["m"]
        mg = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        ys, xs = np.nonzero(mg)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = int(0.08 * max(y1 - y0, x1 - x0)) + 4
        Y0, Y1, X0, X1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        rgba = np.dstack([rgb, mg.astype(np.uint8) * 255])[Y0:Y1, X0:X1]
        crop = os.path.join(out, "crops", f"object_{i}.png")
        Image.fromarray(rgba, "RGBA").save(crop)
        grey = np.where(mg[..., None], rgb, 128).astype(np.uint8)[Y0:Y1, X0:X1]
        tmp = os.path.join(out, "_vlm_inputs", f"grey_{i}.png")  # kept (not deleted): what the VLM was shown
        Image.fromarray(grey).save(tmp)
        name = ask_vlm(tmp, f"This was detected as '{o['phrases'][0]}'. What is this object exactly? "
                            "Answer only with its name in English, as a short lowercase noun phrase of 1 to 4 words "
                            "(for example: truck, pallet of boxes, man in blue vest), no punctuation.", port, max_len=20)
        name = name.strip().splitlines()[0].strip(" .\"'").lower() if name.strip() else o["phrases"][0]

        q = to_floor(Pg[erode(m, 1) if erode(m, 1).sum() > 40 else m])
        q = q[np.isfinite(q).all(1)]
        top = float(np.percentile(q[:, 2], 98))
        base_z = float(max(0.0, np.percentile(q[:, 2], 2)))  # >0 if it sits on something
        if base_z < 0.10 or top < 0.4:
            base_z = 0.0
        xy = q[:, :2]
        c = np.median(xy, 0)
        ev, evec = np.linalg.eigh(np.cov((xy - c).T))
        axis, perp = evec[:, 1], evec[:, 0]
        width_pca = float(np.subtract(*np.percentile(xy @ axis, [95, 5])))
        depth_pca = float(np.subtract(*np.percentile(xy @ perp, [95, 5])))
        # seen width: horizontal extent of the crop in pixels x depth / focal
        zc = float(np.median(Pg[m][:, 2]))
        width_img = (x1 - x0) / 2 * zc / fx
        dist = float(np.linalg.norm(c))  # camera at (0,0) of the floor frame
        to_cam = -c / max(np.linalg.norm(c), 1e-6)
        if width_pca > 1.6 * max(depth_pca, 0.02):  # flat or elongated object: faces perpendicular to its axis
            front = perp if perp @ to_cam > 0 else -perp
            width, depth = max(width_pca, width_img * 0.9), depth_pca
        else:  # round or cubic: the front cannot be told from the shape; assume it faces the camera
            front = to_cam
            width, depth = width_img, max(depth_pca, width_img * 0.5)
        # yaw: a model with its front towards -y, rotated 'yaw' degrees (counterclockwise), ends up facing 'front'
        yaw = (math.degrees(math.atan2(front[1], front[0])) + 90) % 360
        # foot point in the image (bottom center of the crop) for drawing the guides
        res.append(dict(id=i, name=name, type=o["phrases"][0], phrases=o["phrases"],
                        center_m=[round(float(c[0]), 3), round(float(c[1]), 3)], base_z_m=round(base_z, 3),
                        height_m=round(top - base_z, 3), width_m=round(float(width), 3), depth_m=round(float(depth), 3),
                        yaw_deg=round(yaw, 1), front_reliable=bool(width_pca > 1.6 * max(depth_pca, 0.02)),
                        camera_distance_m=round(dist, 3), image_box=[int(x0), int(y0), int(x1), int(y1)],
                        image_foot=[int((x0 + x1) / 2), int(y1)], crop=os.path.relpath(crop, out), pixels=int(m.sum())))
        footprints.append((xy, top, base_z))
        print(f"  object {i}: {name:22s} height {top-base_z:.2f} m  width {width:.2f}  at {dist:.2f} m  faces {yaw:.0f} deg", flush=True)

    # Raised but with nothing under it to hold it up: it stands on the floor and its legs are hidden
    # (chairs in front of a table, thin legs). Something on a counter keeps its height (the counter is below).
    # Only low furniture (top under 1.3 m) and never people: a person or a truck that looks raised may really
    # be standing on a dock or platform that was not detected.
    for k, (xy, top, base_z) in enumerate(footprints):
        if base_z <= 0.0 or top > 1.3 or PERSON.search(res[k]["name"] + " " + res[k]["type"]):
            continue
        c = np.median(xy, 0)
        supported = any(j != k and abs(t2 - base_z) < 0.2 and np.min(np.linalg.norm(xy2 - c, axis=1)) < 0.3
                        for j, (xy2, t2, b2) in enumerate(footprints))
        if not supported:
            res[k]["base_z_m"] = 0.0
            res[k]["height_m"] = round(top, 3)
            res[k]["legs_hidden"] = True
            print(f"  object {res[k]['id']}: raised {base_z:.2f} m with nothing below -> on the floor, height {top:.2f} m", flush=True)

    pairs = []
    for a in range(len(res)):
        for b in range(a + 1, len(res)):
            dd = math.dist(res[a]["center_m"], res[b]["center_m"])
            pairs.append(dict(a=res[a]["id"], b=res[b]["id"], distance_m=round(dd, 3)))
    scene_map = dict(photo=os.path.abspath(photo), W=W, H=H, fov_deg=round(fov_used, 2), fov_source=fov_source,
                     camera_height_m=round(float(d), 3), cv_axes_in_floor=Mx.T.tolist(), fx_px=fx * 2,
                     note="floor frame: camera at (0,0,height); y = where the camera looks; x = right; z = up",
                     objects=res, distances=pairs)
    json.dump(scene_map, open(os.path.join(out, "map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(out, "object_masks.npz"), masks=np.array([o["m"] for o in objects], bool), floor=floor)
    draw(out, rgb, objects, res, pairs)


def draw(out, rgb, objects, res, pairs):
    H, W = rgb.shape[:2]
    rng = np.random.default_rng(7)
    cols = [tuple(int(v) for v in rng.integers(70, 256, 3)) for _ in res]
    base = rgb.astype(float)
    for o, c in zip(objects, cols):
        mg = np.array(Image.fromarray(o["m"].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        edge = mg & ~erode(mg, 3)
        base[mg] = base[mg] * 0.75 + np.array(c) * 0.25
        base[edge] = c
    im = Image.fromarray(base.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    s = max(W, H) / 1280
    f1, f2 = font(int(22 * s)), font(int(16 * s))
    by_id = {r["id"]: r for r in res}
    # to keep it readable: each object is joined only to its 2 nearest neighbours (all distances are in map.json)
    near = set()
    for r in res:
        closest = sorted((p for p in pairs if r["id"] in (p["a"], p["b"])), key=lambda p: p["distance_m"])[:2]
        near.update((p["a"], p["b"]) for p in closest)
    for p in [p for p in pairs if (p["a"], p["b"]) in near]:  # lines between objects with their distance
        A, B = by_id[p["a"]]["image_foot"], by_id[p["b"]]["image_foot"]
        dr.line([tuple(A), tuple(B)], fill=(255, 255, 255), width=max(1, int(2 * s)))
        mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        t = f"{p['distance_m']:.2f} m"
        bb = dr.textbbox((mx, my), t, font=f2, anchor="mm")
        dr.rectangle([bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2], fill=(20, 20, 20))
        dr.text((mx, my), t, font=f2, fill=(255, 255, 255), anchor="mm")
    for r, c in zip(res, cols):
        x0, y0, x1, y1 = r["image_box"]
        t = f"{r['id']}. {r['name']}  {r['height_m']*100:.0f} cm"
        text_w = dr.textlength(t, font=f1)
        x0 = int(min(max(4, x0), W - text_w - 8))
        y0 = max(int(30 * s), y0)
        bb = dr.textbbox((x0, y0 - 4), t, font=f1, anchor="ld")
        dr.rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], fill=c)
        dr.text((x0, y0 - 4), t, font=f1, fill=(0, 0, 0), anchor="ld")
        ax, ay = r["image_foot"]
        dr.ellipse([ax - 6 * s, ay - 6 * s, ax + 6 * s, ay + 6 * s], fill=c, outline=(0, 0, 0))
    im.save(os.path.join(out, "guides.png"))


if __name__ == "__main__":
    main()
