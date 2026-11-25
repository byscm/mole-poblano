# Fits a built room to its photos: puts Blender's camera where each photo was taken, then moves / turns / scales each
# loose piece of furniture until its silhouette matches the SAM 3 mask of that piece in ALL the photos at once.
#   1) cameras: each one moved (x, y, z) and turned (yaw, pitch, roll) a little, plus ONE focal length for the whole
#      video, so the floor, the ceiling and the furniture as a whole line up with the photo;
#   2) furniture: each piece moved (x, y), turned and scaled so its mean overlap (IoU) over every photo that sees it
#      goes up; the worst 20% of photos are ignored (SAM sometimes marks the wrong thing);
#   rounds 1 and 2 are repeated. Renders are flat colour ids (Workbench), small (photo size x SCALE).
# Usage: blender -b --factory-startup -P fit_to_photos.py -- <room.blend> <inventory_map.json> <fit_folder> <items.json>
#   fit_folder/masks/<photo>.npz from photo_masks.py; items.json: [{"key", "objects"|"prefixes", "mask", "fit",
#   "max_move" m (0.5), "max_turn" deg (15), "max_scale" (0.15)}] - built-ins: max_turn 0, max_scale 0, a small max_move
# Writes fit_folder/: scores.json (before / after per piece and per photo), cameras_fit.json, fitted.blend,
#   compare_<photo>.png (photo | before | after) for the photos in COMPARE (env, comma separated) or the 6 best covered.
import bpy, json, math, os, sys, time
import numpy as np
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
BLEND, MAP, FIT, ITEMS = (os.path.abspath(p) for p in a[:4])
SCALE = float(os.environ.get("SCALE", "0.25")); ROUNDS = int(os.environ.get("ROUNDS", "2"))
TRIM = 0.2; MIN_VIS = 0.002
m = json.load(open(MAP, encoding="utf-8")); items = json.load(open(ITEMS, encoding="utf-8"))
cams = [c for c in m["cameras"] if os.path.exists(f"{FIT}/masks/{c['photo']}.npz")]
T0 = time.time()
log = lambda *s: print(f"[{time.time() - T0:6.0f}s]", *s, flush=True)

bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
ORIG = {o.name: o.matrix_world.copy() for o in sc.objects}

# ---- pieces: an empty pivot at the bottom centre of each item's objects
def members(it):
    if "objects" in it: return [sc.objects[n] for n in it["objects"] if n in sc.objects]
    return [o for o in sc.objects if o.type == "MESH" and any(o.name.startswith(p) for p in it["prefixes"])]


def bbox(objs):
    P = np.array([list(o.matrix_world @ Vector(c)) for o in objs for c in o.bound_box]); return P.min(0), P.max(0)


for it in items:
    it["objs"] = members(it)
    lo, hi = bbox(it["objs"]); piv = bpy.data.objects.new("pivot_" + it["key"], None); sc.collection.objects.link(piv)
    piv.location = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]); bpy.context.view_layer.update()
    for o in it["objs"]:
        mw = o.matrix_world.copy(); o.parent = piv; o.matrix_parent_inverse = piv.matrix_world.inverted(); o.matrix_world = mw
    it["pivot"] = piv; it["home"] = piv.matrix_world.copy(); it["p"] = [0.0, 0.0, 0.0, 1.0]  # dx, dy, dyaw_deg, scale
    log(it["key"], len(it["objs"]), "objects")

# ---- flat colour ids: 0 nothing, 1 wall / other, 2 floor, 3 ceiling, 4.. items
LEV = (0, 128, 255); CODES = [(r, g, b) for r in range(3) for g in range(3) for b in range(3)][1:]
lin = lambda s: (s / 255) / 12.92 if s / 255 <= 0.04045 else ((s / 255 + 0.055) / 1.055) ** 2.4
LUT = np.zeros(27, int)


def paint(o, label):
    r, g, b = CODES[label - 1]; o.color = (lin(LEV[r]), lin(LEV[g]), lin(LEV[b]), 1); LUT[r * 9 + g * 3 + b] = label


owner = {}
for k, it in enumerate(items):
    for o in it["objs"]: owner[o.name] = 4 + k
for o in sc.objects:
    if o.type != "MESH": continue
    paint(o, owner.get(o.name, 2 if o.name == "floor" else 3 if o.name == "ceiling" else 1))
    o.hide_render = False
LABEL = {it["key"]: 4 + k for k, it in enumerate(items)}

sc.render.engine = "BLENDER_WORKBENCH"; sh = sc.display.shading
sh.light = "FLAT"; sh.color_type = "OBJECT"; sh.show_object_outline = False; sh.show_shadows = False
sh.show_cavity = False; sh.show_specular_highlight = False; sh.show_xray = False
sc.display.render_aa = "OFF"; sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
sc.view_settings.exposure = 0; sc.view_settings.gamma = 1
sc.world.color = (0, 0, 0); sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"
cd = bpy.data.cameras.new("fitcam"); cam = bpy.data.objects.new("fitcam", cd); sc.collection.objects.link(cam); sc.camera = cam
cd.sensor_fit = "VERTICAL"; cd.sensor_width = cd.sensor_height = 36.0; cd.clip_start = 0.05; cd.clip_end = 50
TMP = os.path.join(FIT, "_id.png")

# ---- cameras: starting pose from the map, plus a correction
for c in cams:
    pm = json.load(open(os.path.join(c["folder"], "map.json"), encoding="utf-8"))
    place = Matrix.Translation((*c["xy_m"], 0)) @ Matrix.Rotation(math.radians(c["turn_deg"]), 4, "Z")
    ex, ey, ez = (Vector(v) for v in pm["cv_axes_in_floor"])
    c["M0"] = place @ Matrix.Translation((0, 0, pm["camera_height_m"] * c.get("shrink", 1.0))) @ Matrix((ex, -ey, -ez)).transposed().to_4x4()
    c["W"], c["H"] = round(pm["W"] * SCALE), round(pm["H"] * SCALE); c["fx0"] = pm["fx_px"]; c["HH"] = pm["H"]
    c["q"] = [0.0] * 6  # dx, dy, dz, dyaw, dpitch, droll (deg)
    d = np.load(f"{FIT}/masks/{c['photo']}.npz"); c["mk"] = {k: d[k] for k in d.files}
    big = [mm for k, v in c["mk"].items() if k not in ("floor", "ceiling") for mm in v if mm.mean() > MIN_VIS]
    c["furn"] = np.any(big, 0) if big else np.zeros((c["H"], c["W"]), bool)
FX = [float(np.median([c["fx0"] for c in cams]))]  # one focal length (pixels at full size) for the whole video


def cam_matrix(c, q=None):
    q = q or c["q"]; M0 = c["M0"]; R0 = M0.to_3x3().to_4x4(); t = M0.translation + Vector(q[:3])
    R = Matrix.Rotation(math.radians(q[3]), 4, "Z") @ R0 @ Matrix.Rotation(math.radians(q[4]), 4, "X") @ Matrix.Rotation(math.radians(q[5]), 4, "Z")
    return Matrix.Translation(t) @ R


def render_ids(c, q=None, fx=None):
    cam.matrix_world = cam_matrix(c, q); cd.lens = (fx or FX[0]) / c["HH"] * 36.0
    sc.render.resolution_x, sc.render.resolution_y = c["W"], c["H"]; sc.render.filepath = TMP
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(TMP, check_existing=False); px = np.empty(c["W"] * c["H"] * 4, np.float32); im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    px = (px.reshape(c["H"], c["W"], 4)[::-1, :, :3] * 255)
    code = np.digitize(px, [64, 192]); return LUT[code[..., 0] * 9 + code[..., 1] * 3 + code[..., 2]]


def iou(x, y):
    u = np.logical_or(x, y).sum(); return float(np.logical_and(x, y).sum() / u) if u else 0.0


def cam_score(c, lab):
    s = [iou(lab == 2, c["mk"]["floor"].any(0)) if len(c["mk"]["floor"]) else None,
         iou(lab == 3, c["mk"]["ceiling"].any(0)) if len(c["mk"]["ceiling"]) else None,
         iou(lab >= 4, c["furn"])]
    s = [v for v in s if v is not None]; return sum(s) / len(s)


def item_iou(c, lab, it):
    ms = c["mk"].get(it["mask"], [])
    r = lab == LABEL[it["key"]]
    return max([iou(r, mm) for mm in ms if mm.mean() > MIN_VIS / 2] or [0.0])


def descend(x, steps, lo, hi, f, levels=3, sweeps=2):
    best = f(x)
    for lvl in range(levels):
        st = [s / 2 ** lvl for s in steps]
        for _ in range(sweeps):
            moved = False
            for i in range(len(x)):
                for sg in (1, -1):
                    y = list(x); y[i] = min(hi[i], max(lo[i], y[i] + sg * st[i]))
                    if y == x: continue
                    v = f(y)
                    if v > best + 1e-4: x, best, moved = y, v, True; break
            if not moved: break
    return x, best


def set_item(it, p):
    it["pivot"].matrix_world = (Matrix.Translation((p[0], p[1], 0)) @ Matrix.Translation(it["home"].translation)
                                @ Matrix.Rotation(math.radians(p[2]), 4, "Z") @ Matrix.Scale(p[3], 4)
                                @ Matrix.Translation(-it["home"].translation) @ it["home"])
    bpy.context.view_layer.update()


def trimmed(v):
    v = sorted(v, reverse=True); n = len(v) - int(len(v) * TRIM) if len(v) >= 5 else len(v); return sum(v[:n]) / n if n else 0.0


def report(tag):
    labs = {c["photo"]: render_ids(c) for c in cams}; out = {"cameras": {}, "items": {}}
    for c in cams: out["cameras"][c["photo"]] = round(cam_score(c, labs[c["photo"]]), 3)
    for it in items:
        per = {c["photo"]: round(item_iou(c, labs[c["photo"]], it), 3) for c in cams if (labs[c["photo"]] == LABEL[it["key"]]).mean() > MIN_VIS}
        out["items"][it["key"]] = dict(score=round(trimmed(list(per.values())), 3), photos=len(per), per_photo=per)
    out["cameras_mean"] = round(float(np.mean(list(out["cameras"].values()))), 3)
    out["items_mean"] = round(float(np.mean([v["score"] for k, v in out["items"].items() if v["photos"]])), 3)
    log(tag, "cameras", out["cameras_mean"], "pieces", out["items_mean"], {k: (v["score"], v["photos"]) for k, v in out["items"].items()})
    return out


t = time.time(); render_ids(cams[0]); log("one render", round(time.time() - t, 3), "s")
scores = {"before": report("BEFORE")}
compare = os.environ.get("COMPARE", "").split(",") if os.environ.get("COMPARE") else None
if not compare:
    cover = sorted(cams, key=lambda c: -sum(1 for it in items if scores["before"]["items"][it["key"]]["per_photo"].get(c["photo"], 0) > 0))
    compare = [c["photo"] for c in cover[:6]]
before_q = {c["photo"]: list(c["q"]) for c in cams}

for rnd in range(ROUNDS):
    # 1) cameras
    for c in cams:
        c["q"], s = descend(c["q"], [0.1, 0.1, 0.1, 3, 3, 2], [-0.5, -0.5, -0.6, -15, -15, -8], [0.5, 0.5, 0.6, 15, 15, 8],
                            lambda q: cam_score(c, render_ids(c, q)), levels=3)
    sub = cams[::3]  # one focal length for the video, judged on a third of the photos
    best_f, best_v = FX[0], -1
    for k in (0.92, 0.96, 1.0, 1.04, 1.08):
        v = float(np.mean([cam_score(c, render_ids(c, fx=FX[0] * k)) for c in sub]))
        if v > best_v: best_f, best_v = FX[0] * k, v
    FX[0] = best_f; log("round", rnd + 1, "cameras done, focal px", round(FX[0]))
    # 2) furniture, each piece over every photo that sees it
    for it in [i for i in items if i.get("fit")]:
        labs = {c["photo"]: render_ids(c) for c in cams}
        see = [c for c in cams if (labs[c["photo"]] == LABEL[it["key"]]).mean() > MIN_VIS]
        if not see: log(it["key"], "seen by no photo"); continue
        def f(p, it=it, see=see):
            set_item(it, p); return trimmed([item_iou(c, render_ids(c), it) for c in see])
        R, T, S = it.get("max_move", 0.5), it.get("max_turn", 15), it.get("max_scale", 0.15)  # common-sense limits
        lo = [-R, -R, -T, 1 - S]; hi = [R, R, T, 1 + S]
        it["p"], s = descend(it["p"], [0.12, 0.12, 5, 0.05], lo, hi, f, levels=3); set_item(it, it["p"])
        log("round", rnd + 1, it["key"], "photos", len(see), "score", round(s, 3), "move", [round(v, 3) for v in it["p"]])
    scores[f"round_{rnd + 1}"] = report(f"ROUND {rnd + 1}")

scores["after"] = scores[f"round_{ROUNDS}"] if ROUNDS else scores["before"]
scores["moves"] = {it["key"]: dict(dx_m=round(it["p"][0], 3), dy_m=round(it["p"][1], 3), turn_deg=round(it["p"][2], 1), scale=round(it["p"][3], 3)) for it in items}
scores["focal_px"] = round(FX[0], 1)
json.dump(scores, open(os.path.join(FIT, "scores.json"), "w"), indent=1)
json.dump({c["photo"]: dict(matrix=[list(r) for r in cam_matrix(c)], fx_px=FX[0], correction=[round(v, 3) for v in c["q"]]) for c in cams},
          open(os.path.join(FIT, "cameras_fit.json"), "w"), indent=1)

# ---- photo | before | after, solid colours over the photo
sh.light = "STUDIO"; sh.color_type = "MATERIAL"; sc.display.render_aa = "8"; sc.view_settings.view_transform = "AgX"
final = {it["key"]: list(it["p"]) for it in items}; fitted_q = {c["photo"]: list(c["q"]) for c in cams}; fx_after = FX[0]
fx_before = float(np.median([c["fx0"] for c in cams]))
for c in [c for c in cams if c["photo"] in compare]:
    c["W"], c["H"] = round(c["W"] / SCALE / 2), round(c["H"] / SCALE / 2)
    ph = np.array(bpy.data.images.load(os.path.join(c["folder"], "photo.png")).pixels[:], np.float32).reshape(c["HH"], -1, 4)
    ph = ph[::2, ::2][: c["H"], : c["W"]]
    panels = [ph]
    for tag, P, q, fx in (("before", {k: [0, 0, 0, 1] for k in final}, before_q[c["photo"]], fx_before), ("after", final, fitted_q[c["photo"]], fx_after)):
        for it in items: set_item(it, P[it["key"]])
        cam.matrix_world = cam_matrix(c, q); cd.lens = fx / c["HH"] * 36.0
        sc.render.resolution_x, sc.render.resolution_y = c["W"], c["H"]; sc.render.filepath = TMP; bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(TMP, check_existing=False); r = np.empty(c["W"] * c["H"] * 4, np.float32); im.pixels.foreach_get(r); bpy.data.images.remove(im)
        r = r.reshape(c["H"], c["W"], 4); ov = ph.copy(); ov[..., :3] = r[..., :3] * 0.55 + ph[..., :3] * 0.45; panels.append(ov)
    both = np.concatenate(panels, 1); im = bpy.data.images.new("cmp", both.shape[1], both.shape[0], alpha=True); im.pixels = both.ravel()
    im.filepath_raw = os.path.join(FIT, f"compare_{c['photo']}.png"); im.file_format = "PNG"; im.save()
for it in items: set_item(it, final[it["key"]])

# ---- the room with the fitted furniture, original look
D = {}
for it in items:
    for o in it["objs"]: D[o.name] = o.matrix_world.copy()
bpy.ops.wm.open_mainfile(filepath=BLEND)
for n, mw in D.items():
    if n in bpy.data.objects: bpy.data.objects[n].matrix_world = mw
for c in cams:
    cdat = bpy.data.cameras.new("photo_" + c["photo"]); co = bpy.data.objects.new("photo_" + c["photo"], cdat)
    bpy.context.scene.collection.objects.link(co); co.matrix_world = cam_matrix(c)
    cdat.sensor_fit = "VERTICAL"; cdat.sensor_width = cdat.sensor_height = 36.0; cdat.lens = FX[0] / c["HH"] * 36.0
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FIT, "fitted.blend"))
log("DONE")
